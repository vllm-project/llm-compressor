import time

from llmcompressor.modifiers.utils.compression_wrapper import ModuleCompressionWrapper

try:
    import transformers
except ImportError as err:
    transformers = None
    transformers_err = err
import torch
import torch.nn as nn

__all__ = ["ALPSWrapper"]


def L0_proj(
    B: torch.Tensor,
    k_spar: int,
    nm_n: int,
    nm_m: int,
    preserve_sparsity_mask: bool,
    init_mask: torch.Tensor,
) -> torch.Tensor:
    totp, num_cout = B.shape
    if nm_n == 0:
        if not preserve_sparsity_mask:
            D = B.reshape(-1)
        else:
            D = (B * init_mask).reshape(-1)
        _, loss_idx = torch.topk(-(D**2), totp * num_cout - k_spar)
        D[loss_idx] = 0
        D = D.reshape(totp, num_cout)
    else:
        new_dim = totp * num_cout / nm_m
        new_dim = int(new_dim)
        k_spar = totp * num_cout * nm_n / nm_m

        if not preserve_sparsity_mask:
            D = B.t().reshape((new_dim, nm_m))
        else:
            D = (B * init_mask).t().reshape((new_dim, nm_m))
        _, loss_idx = torch.topk(-(D**2), nm_m - nm_n, dim=1)
        D = D.scatter(
            src=torch.zeros((new_dim, nm_m - nm_n)).to(B.device),
            dim=1,
            index=loss_idx,
        )
        D = D.reshape(num_cout, totp).t()
    return D


class ALPSWrapper(ModuleCompressionWrapper):
    """
    Runs SparseGPT on a single module that contains no sub-modules

    Lifecycle:
        - add_batch
        - compress
        - free

    :param name: name of module to run compression on
    :param layer: module to run compression on
    """

    def __init__(self, name, layer):
        super().__init__(name=name, layer=layer)

        # for Hessian calculation
        self.register_buffer(
            "XtX", torch.zeros((self.columns, self.columns), device=self.dev)
        )

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        """
        Add a batch of layer input and output data to the Hessian calculation

        :param inp: tensor containing layer input
        :param out: tensor containing layer output
        """
        # TODO: Test Conv layers
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        inp = inp.float()

        self.XtX += inp.matmul(inp.t())

    def compress(
        self,
        sparsity: float,
        prunen: int = 0,
        prunem: int = 0,
        percdamp: float = 0.01,
        rho: float = 0.1,
        max_iter: int = 300,
        update_iter: int = 3,
        switch_iter: int = 30,
        verbose: bool = False,
        preserve_sparsity_mask: bool = False,
    ):
        """
        Run pruning and quantization(if applicable) on the layer up to the target
        sparsity value.

        :param sparsity: target sparsity to reach for layer
        :param prunen: N for N:M pruning
        :param prunem: M for N:M pruning
        :param blocksize: Number of columns to compress in one pass
        :param percdamp: Amount of dampening to apply to H, as a fraction of the
            diagonal norm
        """
        nm_n = prunen
        nm_m = prunem

        W = self.layer.weight.data.clone()
        W = W.float()
        # TODO: Run tests with Conv layers
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()

        init_mask = None
        if preserve_sparsity_mask:
            # compute existing sparsity mask
            init_mask = torch.where(
                W == 0,
                torch.tensor(1, dtype=torch.bool),
                torch.tensor(0, dtype=torch.bool),
            ).t()
            current_sparsity = init_mask.sum() / W.numel()
            if current_sparsity > sparsity:
                raise ValueError(
                    "The target sparsity is lower than the sparsity "
                    "of the base model. Please retry "
                    "after turning preserve_sparsity_mask=False"
                )

        damp1 = percdamp * torch.mean(torch.diag(self.XtX)).item()
        diag = torch.arange(self.XtX.shape[0], device=self.XtX.device)
        self.XtX[diag, diag] += damp1

        X_norm = torch.diag(self.XtX).sqrt() + 1e-8
        if verbose:
            print("The norm is ", X_norm.min(), X_norm.max())
        self.XtX = self.XtX / X_norm
        self.XtX = (self.XtX.T / X_norm).T

        self.YtX = torch.zeros_like(W)
        self.YtX = torch.matmul(W * X_norm, self.XtX)

        admm_st = time.time()

        XTX_inv = torch.zeros_like(self.XtX).float()
        B = (W * X_norm).t().clone()
        W = None
        if verbose:
            B_orig = B.clone()
        V = torch.zeros_like(B)
        D = torch.zeros_like(B)
        D_suppp = torch.zeros_like(B)
        D_supp = torch.zeros_like(B)

        totp, num_cout = B.shape

        L, Q = torch.linalg.eigh(self.XtX.double())

        XTX_inv = (Q @ ((1 / (L + (rho))) * Q).T).float()

        init_rho = False

        if verbose:
            Res0 = self.YtX.T
            Res0 = torch.sum(B_orig * Res0)
            Res0 = torch.sum(Res0)

        params = B.shape[0] * B.shape[1]
        k_spar = int((1 - sparsity) * params)

        D = L0_proj(
            B.clone(),
            k_spar,
            nm_n,
            nm_m,
            preserve_sparsity_mask,
            init_mask,
        )

        D_suppp = (D == 0).to(torch.float).reshape(-1)
        D_init = D.clone()

        for i_admm in range(max_iter):
            B = XTX_inv @ (self.YtX.T - V + rho * D)
            D = (V + rho * B) / rho

            D = L0_proj(
                D,
                k_spar,
                nm_n,
                nm_m,
                preserve_sparsity_mask,
                init_mask,
            )

            V = V + rho * (B - D)

            if (i_admm + 1) % update_iter == 0:
                D_supp = (D.reshape(-1) == 0).to(torch.float)
                supp_change = torch.sum((D_supp - D_suppp) ** 2)

                if supp_change / k_spar > 0.1:
                    init_rho = True
                    rho *= 1.3
                elif supp_change / k_spar > 0.005:
                    init_rho = True
                    rho *= 1.2
                elif supp_change > 0.5:
                    if init_rho:
                        rho *= 1.1
                    else:
                        rho /= 5
                        B = B_orig.clone()
                        D = D_init.clone()
                        V = torch.zeros_like(B)
                else:
                    if init_rho:
                        break
                    else:
                        rho /= 5

                D_suppp = (D_supp).clone()
                if rho > 1e6:
                    rho = 1e6

                XTX_inv = (Q @ ((1 / (L + (rho))) * Q).T).float()

                if i_admm >= switch_iter and supp_change / k_spar < 0.0003:
                    break

        B = L0_proj(
            B,
            k_spar,
            nm_n,
            nm_m,
            preserve_sparsity_mask,
            init_mask,
        )

        V = None
        D = None

        if verbose:
            Res = torch.matmul(self.XtX, B) - self.YtX.T
            Res = torch.diag(torch.matmul((B - B_orig).t(), Res))
            error = torch.sum(Res) / Res0
            error = error.item()

            print("Before backsolve, error is {}".format(error))
        admm_time = time.time() - admm_st
        back_st = time.time()
        B = self.cg_batch(
            (self.XtX),
            self.YtX.T,
            (B != 0).to(torch.float),
            M_bmm=None,
            X0=B,
            rtol=1e-4,
            atol=0.0,
            maxiter=10,
            verbose=verbose,
        )
        back_time = time.time() - back_st
        if verbose:
            Res = torch.matmul(self.XtX, B) - self.YtX.T
            Res = torch.diag(torch.matmul((B - B_orig).t(), Res))
            error = torch.sum(Res) / Res0
            error = error.item()

            print("Number of iter is {}".format(i_admm))
            print("Final Error is {}".format(error))
            print("Time is admm: {} back:{}".format(admm_time, back_time))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # TODO: Run tests with Conv layers
        if isinstance(self.layer, transformers.Conv1D):
            W = (
                (B.t() / X_norm)
                .t()
                .reshape(self.layer.weight.shape)
                .to(self.layer.weight.data.dtype)
            )
        else:
            W = (
                (B.t() / X_norm)
                .reshape(self.layer.weight.shape)
                .to(self.layer.weight.data.dtype)
            )
        # This is a bit hacky, but FSDP updates only work if we change the weight in
        # place, clone() or direct assignment won't work
        # TODO: Run tests for cases with FSDP and how they interact
        self.layer.weight -= self.layer.weight
        self.layer.weight += W

    def cg_batch(
        self,
        A,
        B,
        A_supp,
        M_bmm=None,
        X0=None,
        rtol=1e-3,
        atol=0.0,
        maxiter=None,
        verbose=False,
    ):
        """Solves a batch of PD matrix linear systems using the preconditioned
        CG algorithm.
        This function solves matrix linear systems of the form
        A X = B,where A is a n x n positive definite matrix and B is
        a n x m matrix, and X is the n x m matrix representing the solution for
        the ith system.
        Args:
            A_bmm: A callable that performs a batch matrix multiply of A and
            a n x m matrix.
            B: A n x m matrix representing the right hand sides.
            M_bmm: (optional) A callable that performs a batch matrix multiply of
            the preconditioning matrices M and a n x m matrix. (default=identity matrix)
            X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
            rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
            atol: (optional) Absolute tolerance for norm of residual. (default=0)
            maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
            verbose: (optional) Whether or not to print status messages. (default=False)
        """
        n, m = B.shape

        if M_bmm is None:
            M_bmm = lambda x: x  # noqa: E731
        if X0 is None:
            X0 = M_bmm(B)
        if maxiter is None:
            maxiter = 5 * n

        assert B.shape == (n, m)
        assert X0.shape == (n, m)
        assert rtol > 0 or atol > 0
        assert isinstance(maxiter, int)

        X_k = X0
        R_k = B - A @ X_k
        R_k = R_k * A_supp
        Z_k = M_bmm(R_k)
        P_k = torch.zeros_like(Z_k)

        P_k1 = P_k
        R_k1 = R_k
        R_k2 = R_k
        X_k1 = X0
        Z_k1 = Z_k
        Z_k2 = Z_k

        B_norm = torch.norm(B, dim=1)
        stopping_matrix = torch.max(rtol * B_norm, atol * torch.ones_like(B_norm))

        if verbose:
            print("%03s | %010s %06s" % ("it", "dist", "it/s"))

        optimal = False
        start = time.perf_counter()
        for k in range(1, maxiter + 1):
            Z_k = M_bmm(R_k)

            if k == 1:
                P_k = Z_k
                R_k1 = R_k
                X_k1 = X_k
                Z_k1 = Z_k
            else:
                R_k2 = R_k1
                Z_k2 = Z_k1
                P_k1 = P_k
                R_k1 = R_k
                Z_k1 = Z_k
                X_k1 = X_k
                denominator = (R_k2 * Z_k2).sum(0)
                denominator[denominator == 0] = 1e-8
                beta = (R_k1 * Z_k1).sum(0) / denominator
                P_k = Z_k1 + beta.unsqueeze(0) * P_k1

            denominator = (P_k * (A @ P_k)).sum(0)
            denominator[denominator == 0] = 1e-8
            alpha = (R_k1 * Z_k1).sum(0) / denominator
            X_k = X_k1 + alpha.unsqueeze(0) * P_k
            R_k = R_k1 - alpha.unsqueeze(0) * (A @ P_k)
            R_k = R_k * A_supp
            residual_norm = torch.norm(A @ X_k - B, dim=1)

            if verbose:
                print("%03d | %8.4e" % (k, torch.max(residual_norm / B_norm)))

            if (residual_norm <= stopping_matrix).all():
                optimal = True
                break
        end = time.perf_counter()
        if verbose:
            if optimal:
                print(
                    "Terminated in %d steps (optimal). Took %.3f ms."
                    % (k, (end - start) * 1000)
                )
            else:
                print(
                    "Terminated in %d steps (reached maxiter). Took %.3f ms."
                    % (k, (end - start) * 1000)
                )
        return X_k

    def free(self):
        """
        Free the Hessian memory after the layer is complete
        """
        delattr(self, "XtX")
        delattr(self, "YtX")
        super().free()

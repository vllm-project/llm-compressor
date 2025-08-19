from pydantic import BaseModel, model_validator


class CalibrationConfig(BaseModel):
    moe_calibrate_all_experts: bool
    moe_calibrate_gated_acts: bool

    @model_validator(mode="after")
    def validate_config(self):
        if not self.moe_calibrate_gated_acts and not self.moe_calibrate_all_experts:
            raise NotImplementedError(
                "Using all experts for activations without calibrating all experts is not supported. "
                "Please set moe_calibrate_gated_acts=True or moe_calibrate_all_experts=True."
            )

        return self

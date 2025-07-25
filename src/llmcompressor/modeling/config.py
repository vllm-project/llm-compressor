from pydantic import BaseModel, model_validator


class CalibrationConfig(BaseModel):
    moe_calibrate_all_experts: bool
    moe_calibrate_gated_acts: bool

    @model_validator(mode="after")
    def validate_config(self):

        if not self.moe_calibrate_gated_acts and not self.moe_calibrate_all_experts:
            raise NotImplementedError(
                "At least one of moe_calibrate_gated_acts or "
                "moe_calibrate_all_experts must be set to True."
            )

        return self

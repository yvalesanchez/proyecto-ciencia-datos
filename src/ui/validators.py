from dataclasses import dataclass


@dataclass
class NumericFieldValidation:
    value: float
    error: str | None = None


def parse_float(value: str, field_name: str) -> NumericFieldValidation:
    """
    Convierte un string a float, devolviendo un posible mensaje de error.
    Se usa para validar campos numéricos en la interfaz gráfica.
    """
    value = value.strip()
    if value == "":
        return NumericFieldValidation(
            value=0.0,
            error=f"El campo '{field_name}' no puede estar vacío.",
        )

    try:
        num = float(value)
        if num < 0:
            return NumericFieldValidation(
                value=num,
                error=f"El campo '{field_name}' no puede ser negativo.",
            )
        return NumericFieldValidation(value=num, error=None)
    except ValueError:
        return NumericFieldValidation(
            value=0.0,
            error=f"El campo '{field_name}' debe ser un número válido.",
        )
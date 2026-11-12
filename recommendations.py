def generate_suggestions(patient, prediction):
    suggestions = []

    if prediction == 1:
        result = "Cardiovascular Disease Indicator Detected"

        suggestions.append(
            "Consult a qualified healthcare professional "
            "for further cardiovascular assessment."
        )

    else:
        result = "No Cardiovascular Disease Indicator Detected"

        suggestions.append(
            "Continue maintaining a heart-healthy lifestyle "
            "and attend routine health checkups."
        )

    systolic = patient.get("ap_hi")

    if systolic is not None and systolic >= 140:
        suggestions.append(
            "Your systolic blood pressure is elevated. "
            "Arrange a medical assessment."
        )

    cholesterol = patient.get("cholesterol")

    if cholesterol == 3:
        suggestions.append(
            "Your cholesterol category is high. "
            "Discuss cholesterol testing and management with a doctor."
        )

    glucose = patient.get("gluc")

    if glucose == 3:
        suggestions.append(
            "Your glucose category is high. "
            "Discuss appropriate blood glucose testing with a doctor."
        )

    bmi = patient.get("bmi")

    if bmi is not None and bmi >= 30:
        suggestions.append(
            "Your BMI is in the obesity range. "
            "Discuss a suitable weight-management plan."
        )

    return result, suggestions

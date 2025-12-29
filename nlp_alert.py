from datetime import datetime

def generate_alert(object_name):
    """
    Generates a professional alert message when an object is removed.
    """

    alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    alert_message = (
        f"ALERT: The object '{object_name}' was removed at "
        f"{alert_time}. Please investigate immediately."
    )

    return alert_message

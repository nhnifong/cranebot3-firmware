import qrcode
import io

def print_qr_code_ascii(data_string: str):
    """
    Generates a QR code from the given string and prints it to the console
    using ASCII characters.

    Args:
        data_string: The string to encode into the QR code.
    """
    if not isinstance(data_string, str) or not data_string:
        print("Error: A non-empty string must be provided.")
        return

    # The qrcode.QRCode object allows for more control than the basic make() function.
    # We can specify version, error correction, box size, and border.
    # For ASCII output, a smaller box_size and border is often better.
    qr = qrcode.QRCode(
        version=1,  # Controls the size of the QR Code
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1, # The size of each "box" in the QR code grid
        border=4,   # The thickness of the border
    )

    # Add the data to the QR code
    qr.add_data(data_string)
    qr.make(fit=True)

    # The print_ascii method is what does the magic.
    # We can also capture the string output if we wanted to use it elsewhere.
    # For example, by passing an 'out' parameter like a StringIO object.
    print(f"--- QR Code for: '{data_string}' ---")
    qr.print_ascii()
    print("------------------------------------")


# --- Example Usage ---
if __name__ == "__main__":
    # The string from your example
    connection_string = "connect to https://192.168.1.240:4443 on your phone"
    
    # We only encode the URL part for the QR code, as that's the scannable data.
    url_to_encode = "https://192.168.1.240:4443"

    print_qr_code_ascii(url_to_encode)
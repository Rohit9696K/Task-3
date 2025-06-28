from PIL import Image
import os


def encrypt_image(image_path, key):
    image = Image.open(image_path).convert('RGB')  # ensure 3 channels only
    encrypted = Image.new('RGB', image.size)
    width, height = image.size

    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))
            # Encrypt by shifting RGB values
            r = (r + key) % 256
            g = (g + key) % 256
            b = (b + key) % 256
            encrypted.putpixel((x, y), (r, g, b))

    encrypted.save("encrypted_image.png")
    print("✅ Image encrypted and saved as 'encrypted_image.png'.")


def decrypt_image(image_path, key):
    image = Image.open(image_path).convert('RGB')  # ensure 3 channels only
    decrypted = Image.new('RGB', image.size)
    width, height = image.size

    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))
            # Decrypt by reversing the shift
            r = (r - key) % 256
            g = (g - key) % 256
            b = (b - key) % 256
            decrypted.putpixel((x, y), (r, g, b))

    decrypted.save("decrypted_image.png")
    print("✅ Image decrypted and saved as 'decrypted_image.png'.")


def main():
    print("=== Simple Image Encryption Tool ===")
    choice = input("Do you want to (e)ncrypt or (d)ecrypt an image? ").strip().lower()

    if choice not in ['e', 'd']:
        print("❌ Invalid choice. Please enter 'e' or 'd'.")
        return

    image_path = input("Enter the path to the image file: ").strip()
    if not os.path.exists(image_path):
        print("❌ File not found.")
        return

    try:
        key = int(input("Enter an integer key for encryption/decryption: "))
    except ValueError:
        print("❌ Invalid key. Please enter an integer.")
        return

    if choice == 'e':
        encrypt_image(image_path, key)
    else:
        decrypt_image(image_path, key)


if __name__ == "__main__":
    main()

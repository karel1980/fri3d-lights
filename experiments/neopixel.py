import time
from rpi_ws281x import PixelStrip, Color

# LED strip configuration:
LED_COUNT = 10        # Number of LED pixels.
LED_PIN = 18          # GPIO pin connected to the pixels (18 uses PWM!).
LED_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
LED_DMA = 10          # DMA channel to use for generating signal (try 10)
LED_BRIGHTNESS = 255  # Set to 0 for darkest and 255 for brightest
LED_INVERT = False    # True to invert the signal (when using NPN transistor level shift)

# Create NeoPixel object with the specified configuration
strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS)

# Intialize the library (must be called once before other functions).
strip.begin()

# Function to set all pixels to a specific color
def set_all_pixels(strip, color):
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, color)
    strip.show()

# Main loop
if __name__ == "__main__":
    try:
        while True:
            # Set all pixels to red
            set_all_pixels(strip, Color(255, 0, 0))
            time.sleep(1)  # Wait for 1 second

            # Set all pixels to green
            set_all_pixels(strip, Color(0, 255, 0))
            time.sleep(1)  # Wait for 1 second

            # Set all pixels to blue
            set_all_pixels(strip, Color(0, 0, 255))
            time.sleep(1)  # Wait for 1 second

    except KeyboardInterrupt:
        # Clear the NeoPixels when the program is interrupted
        set_all_pixels(strip, Color(0, 0, 0))


import time
from rpi_ws281x import PixelStrip, Color


class LedStrip:
    def __init__(self, count=50, brightness = 55):
        self.count = count

        LED_PIN = 18          # GPIO pin connected to the pixels (18 uses PWM!).
        LED_FREQ_HZ = 800000  # LED signal frequency in hertz (usually 800khz)
        LED_DMA = 10          # DMA channel to use for generating signal (try 10)
        LED_INVERT = False    # True to invert the signal (when using NPN transistor level shift)

        # Create NeoPixel object with the specified configuration
        self.strip = PixelStrip(count, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, brightness)

        # Intialize the library (must be called once before other functions).
        self.strip.begin()

    # Function to set all pixels to a specific color
    def set_all_pixels(self, color):
        for i in range(self.strip.numPixels()):
            self.strip.setPixelColor(i, color)
        self.strip.show()

    def set_array(self, colors):
        if len(colors) != self.strip.numPixels():
            print("Warning: set_array called with wrong number of colors")
        for i in range(max(len(colors),self.count)):
            self.strip.setPixelColor(i, Color(*colors[i]))
        self.strip.show()


# Main loop
if __name__ == "__main__":
    lights = Lights()
    try:
        while True:
            # Set all pixels to red
            lights.set_all_pixels(Color(255, 0, 0))
            time.sleep(1)  # Wait for 1 second

            # Set all pixels to green
            lights.set_all_pixels(Color(0, 255, 0))
            time.sleep(1)  # Wait for 1 second

            # Set all pixels to blue
            lights.set_all_pixels(Color(0, 0, 255))
            time.sleep(1)  # Wait for 1 second

    except KeyboardInterrupt:
        # Clear the NeoPixels when the program is interrupted
        lights.set_all_pixels(Color(0, 0, 0))


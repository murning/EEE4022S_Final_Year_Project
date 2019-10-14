import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib
from time import sleep


class Motor:
    def __init__(self):

        self.GPIO_pins = (5, 6, 13)  # Microstep Resolution MS1-MS3 -> GPIO Pin
        self.direction_pin = 26  # Direction -> GPIO Pin
        self.step_pin = 19  # Step -> GPIO Pin
        self.motor = RpiMotorLib.A4988Nema(self.direction_pin, self.step_pin, self.GPIO_pins, "A4988")
        self.green_LED_pin = 22
        self.red_LED_pin = 27
        GPIO.setmode(GPIO.BCM)

        GPIO.setup(self.green_LED_pin, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.red_LED_pin, GPIO.OUT, initial=GPIO.LOW)

    def rotate(self, degrees, direction):

        amount_steps = int(degrees) / 1.8

        rotation_dir = True

        if direction == "ccw":
            rotation_dir = True
        elif direction == "cw":
            rotation_dir = False
        else:
            print("error")

        self.motor.motor_go(rotation_dir, "Full", int(amount_steps), .015, False, .05)

    def green_led(self):

        GPIO.output(self.green_LED_pin, GPIO.HIGH)  # Turn on
        sleep(3)  # Sleep for 1 second
        GPIO.output(self.green_LED_pin, GPIO.LOW)  # Turn off

    def red_led_on(self):
        GPIO.output(self.red_LED_pin, GPIO.HIGH)  # Turn on

    def red_led_off(self):
        GPIO.output(self.red_LED_pin, GPIO.LOW)  # Turn on

    def clean_up(self):
        GPIO.cleanup()

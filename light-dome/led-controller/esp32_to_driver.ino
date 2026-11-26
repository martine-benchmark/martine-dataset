#include <Adafruit_MCP23X17.h>

#define NUM_BOARDS 4
Adafruit_MCP23X17 mcps[NUM_BOARDS];
uint8_t addresses[NUM_BOARDS] = {0x20, 0x21, 0x22, 0x23};

void setup() {
  Serial.begin(115200);
  delay(3000);
  while (!Serial);
  Serial.println("Multi-Board Solenoid Driver - Interactive Mode");
  Serial.println("Enter a number from 0 to 31 to toggle a solenoid.");
  Serial.println("Enter -1 to turn all solenoids OFF.");
  Serial.println("Press Enter after typing the number.");

  for (int i = 0; i < NUM_BOARDS; i++) {
    if (!mcps[i].begin_I2C(addresses[i])) {
      Serial.print("Couldn't find MCP23017 at address 0x");
      Serial.println(addresses[i], HEX);
      while (1);
    }
  }

  for (int i = 0; i < NUM_BOARDS; i++) {
    for (uint8_t pin = 0; pin < 8; pin++) {
      mcps[i].pinMode(pin, OUTPUT);
      mcps[i].digitalWrite(pin, LOW);
    }
  }

  Serial.println("Found all MCP23017s. Waiting for a number...");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) {
      int channelNumber = input.toInt();

      if (channelNumber >= 0 && channelNumber < 32) {
        int boardIndex = channelNumber / 8;
        int pinNumber = channelNumber % 8;

        bool currentState = mcps[boardIndex].digitalRead(pinNumber);

        if (currentState == LOW) {
          sleep(0.1);
          mcps[boardIndex].digitalWrite(pinNumber, HIGH);
          //Serial.println(pinNumber);
          //Serial.println(" ON !");
          Serial.println("done");
        } else {
          sleep(0.1);
          mcps[boardIndex].digitalWrite(pinNumber, LOW);
          //Serial.println(pinNumber);
          //Serial.println(" OFF !");
          Serial.println("done");
        }
      } else if (channelNumber == -1) {
        // Turn off all lights
        Serial.println("Turning all solenoids OFF.");
        for (int i = 0; i < NUM_BOARDS; i++) {
          for (uint8_t pin = 0; pin < 8; pin++) {
            mcps[i].digitalWrite(pin, LOW);
          }
        }
        Serial.println("done");
      } else {
        Serial.println("Invalid number. Please enter a number from 0 to 31 or -1.");
      }
    }
  }
}

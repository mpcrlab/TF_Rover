void setup() {
  // put your setup code here, to run once:
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(7, OUTPUT);
  pinMode(8, OUTPUT);
  pinMode(9, OUTPUT);
}

void loop() {

    
  // Green light, green, gray, and white wires, green tape
 digitalWrite(9, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(6000);                       // wait for a second
  digitalWrite(9, LOW);    // turn the LED off by making the voltage LOW
  delay(300);                       // wait for a second

  
  // Yellow light, yellow and blue wires, navy blue tape
 digitalWrite(8, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(2000);                       // wait for a second
  digitalWrite(8, LOW);    // turn the LED off by making the voltage LOW
  delay(300);                       // wait for a second */

  
  // Red light, red, orange, and purple wires, red and orange tape
 digitalWrite(7, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(6000);                       // wait for a second
  digitalWrite(7, LOW);    // turn the LED off by making the voltage LOW
  delay(300);                       // wait for a second
}

/**
 * Api.java
 * 
 * Java interface to Flask Server -> To run, use the following steps:
 *      - Start Flask Server (via a call to `python api.py`). This starts a server
 *        at localhost:5000 (http://127.0.0.1:5000)
 *      - Run Api.java => Drops into a prompt to enter NLP commands.
 */      
import java.io.*;
import java.net.*;
import java.util.Arrays;
import java.util.Scanner;


public class Api {
        public static void main(String[] args) throws Exception {
        Scanner reader = new Scanner(System.in);
        while (true) {
            System.out.print("Enter a Natural Language Command: ");
            String command = reader.nextLine();

            URL commandURL = new URL("http://127.0.0.1:5000/model?command=" +
                                     URLEncoder.encode(command, "UTF-8"));
            BufferedReader in = new BufferedReader(new InputStreamReader(commandURL.openStream()));
            String[] response = in.readLine().split(" ");
            String level = response[0];
            String rewardFunction = String.join(" ", Arrays.copyOfRange(response, 1, response.length));
            System.out.println("Level: " + level + " RF: " + rewardFunction);
        }
    }
}  
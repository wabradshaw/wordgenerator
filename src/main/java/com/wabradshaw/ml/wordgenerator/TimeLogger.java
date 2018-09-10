package com.wabradshaw.ml.wordgenerator;

import java.time.Duration;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class TimeLogger {

    public static void printPredictedEndpoint(LocalDateTime start, int epoch, int epochs) {
        LocalDateTime now = LocalDateTime.now();
        Duration taken = Duration.between(start, now);
        double timePerEpoch = taken.getSeconds() * 1.0 / (epoch + 1);
        double secondsLeft = (epochs - epoch) * timePerEpoch;
        String end = now.plusSeconds((long)secondsLeft).format(DateTimeFormatter.ISO_DATE_TIME);
        System.out.println("Predicted end: " + end);
        System.out.println(" ---------------------------------\n");
    }

    public static void printDuration(Duration duration) {
        String output = "Training took ";
        long seconds = duration.getSeconds();
        if(seconds > 7200){
            output = output + (seconds / 3600) + " hours";
        } else if (duration.getSeconds() > 120){
            output = output + (seconds / 60) + " minutes";
        } else {
            output = output + duration.getSeconds() + " seconds";
        }
        System.out.println(output);
    }
}

package com.wabradshaw.ml.wordgenerator;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Main which allows you to train a neural network, including generating samples, and saves the output as a zip file
 */
public class TrainingMain {

    private static final TokenSet TOKEN_SET = TokenSet.PHONEMES_ARPABET_SEPARATE_STRESSES;
    private static final int MAX_WORD_LENGTH = 19;

    private static final int LAYER_SIZE = 200;
    private static final double LEARNING_RATE = 0.2;

    private static final int EPOCHS = 10;
    private static final int BATCHES = 25;
    private static final int BATCH_SIZE = 5350;
    private static final int SAMPLES = 10;
    private static final int SAMPLE_FREQUENCY = 1;

    private static final int SEED = 1234;

    private static final String OUTPUT_FILENAME = "src/main/resources/generatedModelPhonemes";
    private static final String EXISTING_NETWORK_FILENAME = null; //"src/main/resources/generatedModelV2x600";

    public static void main(String[] args) throws Exception {
        NetworkConfiguration config = new NetworkConfiguration(TOKEN_SET, LAYER_SIZE, LEARNING_RATE, SEED);

        MultiLayerNetwork network;
        if(EXISTING_NETWORK_FILENAME == null) {
            network = config.createNetwork();
        } else {
            network = ModelSerializer.restoreMultiLayerNetwork(EXISTING_NETWORK_FILENAME + ".zip");
        }
        network.setListeners(new ScoreIterationListener(1));

        DataSetGenerator dataSetGenerator = new DataSetGenerator(TOKEN_SET);
        WordGenerator wordGenerator = new WordGenerator(TOKEN_SET);

        LocalDateTime start = LocalDateTime.now();

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            for(int batch = 0; batch < BATCHES; batch++) {
                DataSet dataSet = dataSetGenerator.getDataSet(batch, BATCH_SIZE, MAX_WORD_LENGTH);
                dataSet.shuffle(SEED);
                network.fit(dataSet);
            }

            if(epoch % SAMPLE_FREQUENCY == 0) {
                System.out.println("\n -- " + epoch + " --------------------------");
                wordGenerator.generate(SAMPLES, network).forEach(System.out::println);
                System.out.println(" ---------------------------------");
                printPredictedEndpoint(start, epoch, EPOCHS);
                System.out.println(" ---------------------------------\n");
            }
        }

        LocalDateTime end = LocalDateTime.now();

        System.out.println("\n -- FINAL --------------------------");
        wordGenerator.generate(SAMPLES, network).forEach(System.out::println);
        System.out.println(" -----------------------------------");


        printDuration(Duration.between(start, end));

        File locationToSave = new File(OUTPUT_FILENAME + ".zip");
        ModelSerializer.writeModel(network, locationToSave, true);

        System.out.println("DONE");
    }

    private static void printPredictedEndpoint(LocalDateTime start, int epoch, int epochs) {
        LocalDateTime now = LocalDateTime.now();
        Duration taken = Duration.between(start, now);
        double timePerEpoch = taken.getSeconds() * 1.0 / (epoch + 1);
        double secondsLeft = (epochs - epoch) * timePerEpoch;
        String end = now.plusSeconds((long)secondsLeft).format(DateTimeFormatter.ISO_DATE_TIME);
        System.out.println("Predicted end: " + end);
    }

    private static void printDuration(Duration duration) {
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

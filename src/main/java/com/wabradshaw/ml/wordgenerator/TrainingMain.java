package com.wabradshaw.ml.wordgenerator;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.Random;

/**
 * Main which allows you to train a neural network, including generating samples, and saves the output as a zip file
 */
public class TrainingMain {

    private static final TokenSet TOKEN_SET = TokenSet.PHONEMES_ARPABET_SEPARATE_STRESSES;
    private static final int MAX_WORD_LENGTH = 19;

    private static final int LAYER_SIZE = 400;
    private static final double LEARNING_RATE = 0.2;

    private static final int EPOCHS = 10;
    private static final int BATCHES = 50;
    private static final int BATCH_SIZE = 2675;
    private static final int SAMPLES = 10;
    private static final int SAMPLE_FREQUENCY = 50;

    private static final int SEED = 1234;

    private static final String OUTPUT_FILENAME = "src/main/resources/generatedModelPhonemes";
    private static final String EXISTING_NETWORK_FILENAME = null;

    private static final WordGenerator wordGenerator = new WordGenerator(TOKEN_SET);
    private static final Random random = new Random(SEED);

    public static void main(String[] args) throws Exception {

        MultiLayerNetwork network = getNetwork();
        DataSetGenerator dataSetGenerator = new DataSetGenerator(TOKEN_SET);

        LocalDateTime startTime = LocalDateTime.now();

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            trainNetwork(network, dataSetGenerator);

            if(epoch % SAMPLE_FREQUENCY == 0) {
                printSamples(network, Integer.toString(epoch));
                TimeLogger.printPredictedEndpoint(startTime, epoch, EPOCHS);
            }
        }

        LocalDateTime endTime = LocalDateTime.now();

        printSamples(network, "FINAL");
        TimeLogger.printDuration(Duration.between(startTime, endTime));

        saveNetwork(network);

        System.out.println("DONE");
    }

    /**
     * Gets the Neural Network to use. If a file name has been supplied, then an existing network will be loaded and
     * used. Otherwise, the system will create a new neural network.
     *
     * @return A neural network.
     * @throws IOException
     */
    private static MultiLayerNetwork getNetwork() throws IOException {

        MultiLayerNetwork network;
        if(EXISTING_NETWORK_FILENAME == null) {
            NetworkConfiguration config = new NetworkConfiguration(TOKEN_SET, LAYER_SIZE, LEARNING_RATE, SEED);
            network = config.createNetwork();
        } else {
            network = ModelSerializer.restoreMultiLayerNetwork(EXISTING_NETWORK_FILENAME + ".zip");
        }
        network.setListeners(new ScoreIterationListener(1));
        return network;
    }

    /**
     * Train the network using DataSets from the supplied dataSetGenerator.
     *
     * @param network          The network being trained.
     * @param dataSetGenerator The generator to use to produce training data sets.
     */
    private static void trainNetwork(MultiLayerNetwork network, DataSetGenerator dataSetGenerator) {
        for(int batch = 0; batch < BATCHES; batch++) {
            DataSet dataSet = dataSetGenerator.getDataSet(batch, BATCH_SIZE, MAX_WORD_LENGTH);

            // dataSet.shuffle creates a new Random from the input seed, causing each batch/epoch to be shuffled the
            // same way. This approach ensures separate runs with the same seed will be the same, but batches/epochs
            // will be shuffled differently each time.
            dataSet.shuffle(random.nextInt());
            network.fit(dataSet);
        }
    }

    /**
     * Print a series of example words.
     *
     * @param network The network to do the generation.
     * @param label   The name for this series. Typically the name of the training epoch.
     */
    private static void printSamples(MultiLayerNetwork network, String label) {
        System.out.println("\n -- " + label + " --------------------------");
        wordGenerator.generate(SAMPLES, network).forEach(System.out::println);
        System.out.println(" -----------------------------------");
    }

    /**
     * Serialises the network to disk so that it can either be used for generation, or trained further.
     *
     * @param network The neural net being trained.
     * @throws IOException
     */
    private static void saveNetwork(MultiLayerNetwork network) throws IOException {
        File locationToSave = new File(OUTPUT_FILENAME + ".zip");
        ModelSerializer.writeModel(network, locationToSave, true);
    }

}

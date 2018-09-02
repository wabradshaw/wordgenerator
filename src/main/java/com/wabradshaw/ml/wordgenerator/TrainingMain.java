package com.wabradshaw.ml.wordgenerator;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Main which allows you to train a neural network, including generating samples, and saves the output as a
 */
public class TrainingMain {
    public static void main(String[] args) throws Exception {

        NetworkConfiguration config = new NetworkConfiguration(Mode.LEXEMES,
                                                               NetworkConfiguration.CHARS_EN_CAPS_WITH_COMMON,
                                                               200,
                                                               1234);

        MultiLayerNetwork network = config.createNetwork();

        // Load file
        // Split file
        // Tokenize
        DataSet tokens = config.getTokeniser().getTokens();

        // Create iterator
        // Train
        // Examples
        // Save Model
        System.out.println("TODO");
    }
}

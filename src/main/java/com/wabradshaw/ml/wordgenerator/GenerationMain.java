package com.wabradshaw.ml.wordgenerator;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

/**
 * Main which allows you to produce words from the neural network.
 */
public class GenerationMain {

    private static final TokenSet TOKEN_SET = TokenSet.CHARS_EN_CAPS_WITH_COMMON;

    private static final String FILENAME = "src/main/resources/generatedModelV2x1200";
    private static final int WORDS = 300;
    private static final double MIN_LETTER_THRESHOLD = 0.01;

    public static void main(String[] args) throws Exception {

        WordGenerator wordGenerator = new WordGenerator(TOKEN_SET, MIN_LETTER_THRESHOLD);

        MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(FILENAME + ".zip");

        wordGenerator.generate(WORDS, network).forEach(System.out::println);
    }
}
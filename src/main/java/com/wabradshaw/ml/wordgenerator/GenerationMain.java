package com.wabradshaw.ml.wordgenerator;

import com.wabradshaw.ml.wordgenerator.tokenisation.LexemeTokeniser;
import com.wabradshaw.ml.wordgenerator.tokenisation.Tokeniser;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.Random;

/**
 * Main which allows you to produce words from the neural network.
 */
public class GenerationMain {

    private static final Mode MODE = Mode.LEXEMES;
    private static final String[] TOKEN_SET = NetworkConfiguration.CHARS_EN_CAPS_WITH_COMMON;

    private static final String FILENAME = "src/main/resources/generatedModel";
    private static final int WORDS = 30;

    public static void main(String[] args) throws Exception {

        MultiLayerNetwork restored = ModelSerializer.restoreMultiLayerNetwork(FILENAME + ".zip");

        TrainingMain.printSamples(WORDS, TOKEN_SET.length, restored, new LexemeTokeniser(TOKEN_SET));
    }
}
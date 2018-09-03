package com.wabradshaw.ml.wordgenerator;

import com.wabradshaw.ml.wordgenerator.tokenisation.Tokeniser;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;
import java.time.temporal.TemporalUnit;
import java.util.Random;

/**
 * Main which allows you to train a neural network, including generating samples, and saves the output as a
 */
public class TrainingMain {

    private static final Mode MODE = Mode.LEXEMES;
    private static final String[] TOKEN_SET = NetworkConfiguration.CHARS_EN_CAPS_WITH_COMMON;

    private static final int LAYER_SIZE = 200;
    private static final double LEARNING_RATE = 0.2;

    private static final int EPOCHS = 500;
    private static final int SAMPLES = 10;
    private static final int SAMPLE_FREQUENCY = 10;
    private static final int SEED = 1234;

    private static final String FILENAME = "src/main/resources/generatedModel";

    public static void main(String[] args) throws Exception {

        LocalDateTime start = LocalDateTime.now();

        NetworkConfiguration config = new NetworkConfiguration(MODE, TOKEN_SET, LAYER_SIZE, LEARNING_RATE, SEED);

        MultiLayerNetwork network = config.createNetwork();

        DataSet dataSet = config.getTokeniser().getTokens();
        dataSet.shuffle(SEED);

        for (int epoch = 0; epoch <= EPOCHS; epoch++) {
            network.fit(dataSet);

            if(epoch % SAMPLE_FREQUENCY == 0) {
                System.out.println("\n -- " + epoch + " --------------------------");
                printSamples(SAMPLES, TOKEN_SET.length, network, config.getTokeniser());
                System.out.println(" ---------------------------------\n");
            }
        }

        LocalDateTime end = LocalDateTime.now();
        printDuration(Duration.between(start, end));

        File locationToSave = new File(FILENAME + ".zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        ModelSerializer.writeModel(network, locationToSave, false);

        // TODO - Save Model

        System.out.println("DONE");
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

    public static void printSamples(int sampleCount, int possibleTokenCount, MultiLayerNetwork network, Tokeniser tokeniser) {
        INDArray initializationInput = Nd4j.zeros(sampleCount, possibleTokenCount, 1);
        for(int i = 0; i < sampleCount; i++){
            initializationInput.putScalar(new int[]{i, 0, 0}, 1.0);
        }

        network.rnnClearPreviousState();
        INDArray output = network.rnnTimeStep(initializationInput);

        Random rng = new Random();
        StringBuilder[] stringBuilders = new StringBuilder[sampleCount];
        for(int i = 0 ; i < sampleCount; i++){
            stringBuilders[i] = new StringBuilder();
        }

        for( int charId=0; charId<30; charId++ ){
            //Set up next input (single time step) by sampling from previous output
            INDArray nextInput = Nd4j.zeros(sampleCount, possibleTokenCount, 1);

            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            for(int sampleId = 0; sampleId < sampleCount; sampleId++) {
                double[] outputProbDistribution = new double[possibleTokenCount];

                for( int possibleCharId=0; possibleCharId < outputProbDistribution.length; possibleCharId++ ) {
                    outputProbDistribution[possibleCharId] = output.getDouble(sampleId, possibleCharId, 0);
                }

                int sampledCharacterId = sampleFromDistribution(outputProbDistribution, rng);

                stringBuilders[sampleId].append(tokeniser.getToken(sampledCharacterId));

                nextInput.putScalar(new int[]{sampleId, sampledCharacterId, 0}, 1.0);        //Prepare next time step input
            }

            output = network.rnnTimeStep(nextInput);	//Do one time step of forward pass
        }

        String endToken = "$";

        // Print the result
        for(int i = 0 ; i < sampleCount; i++){
            String sample = stringBuilders[i].toString();
            int length = sample.indexOf(endToken);
            if(length > 0){
                System.out.println(sample.substring(0, length));
            } else {
                System.out.println(sample);
            }

        }

    }

    /** Given a probability distribution over discrete classes, sample from the distribution
     * and return the generated class index.
     * @param distribution Probability distribution over classes. Must sum to 1.0
     */
    public static int sampleFromDistribution( double[] distribution, Random rng ){
        double d = 0.0;
        double sum = 0.0;
        for( int t=0; t<10; t++ ) {
            d = rng.nextDouble();
            sum = 0.0;
            for( int i=0; i<distribution.length; i++ ){
                sum += distribution[i];
                if( d <= sum ) return i;
            }
            //If we haven't found the right index yet, maybe the sum is slightly
            //lower than 1 due to rounding error, so try again.
        }
        //Should be extremely unlikely to happen if distribution is a valid probability distribution
        throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
    }
}

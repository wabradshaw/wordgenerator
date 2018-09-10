package com.wabradshaw.ml.wordgenerator;

import com.wabradshaw.ml.wordgenerator.tokenisation.Tokeniser;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.DoubleStream;

public class WordGenerator {

    private final Tokeniser tokeniser;
    private final int possibleTokenCount;

    private final double minDistribution;

    public WordGenerator(TokenSet tokenSet) {
        this(tokenSet, 0.0);
    }

    public WordGenerator(TokenSet tokenSet, double minDistribution) {
        this.tokeniser = tokenSet.getTokeniser();
        this.possibleTokenCount = tokenSet.getLength();
        this.minDistribution = minDistribution;
    }

    public List<String> generate(int sampleCount, MultiLayerNetwork network) {
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

        for( int charId=0; charId<possibleTokenCount; charId++ ){
            //Set up next input (single time step) by sampling from previous output
            INDArray nextInput = Nd4j.zeros(sampleCount, possibleTokenCount, 1);

            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            for(int sampleId = 0; sampleId < sampleCount; sampleId++) {
                double[] outputProbDistribution = new double[possibleTokenCount];

                for( int possibleCharId=0; possibleCharId < outputProbDistribution.length; possibleCharId++ ) {
                    outputProbDistribution[possibleCharId] = output.getDouble(sampleId, possibleCharId, 0);
                }

                int sampledCharacterId = sampleFromDistribution(outputProbDistribution, rng);

                stringBuilders[sampleId].append(tokeniser.toSymbol(sampledCharacterId));

                nextInput.putScalar(new int[]{sampleId, sampledCharacterId, 0}, 1.0);        //Prepare next time step input
            }

            output = network.rnnTimeStep(nextInput);	//Do one time step of forward pass
        }

        List<String> result = new ArrayList<>();
        for(int i = 0 ; i < sampleCount; i++){
            String sample = stringBuilders[i].toString();
            int length = sample.indexOf(TokenSet.EOF_TOKEN);
            if(length > 0){
                result.add(sample.substring(0, length));
            } else {
                result.add(sample);
            }
        }
        return result;

    }

    /** Given a probability distribution over discrete classes, sample from the distribution
     * and return the generated class index.
     * @param distribution Probability distribution over classes. Must sum to 1.0
     */
    public int sampleFromDistribution( double[] distribution, Random rng ){
        double[] likelyDistribution = DoubleStream.of(distribution).map(x -> x >= minDistribution ? x : 0.0).toArray();
        double total = DoubleStream.of(likelyDistribution).sum();

        double target = 0.0;
        double runningTotal = 0.0;

        target = rng.nextDouble()*total;
        runningTotal = 0.0;
        for( int i=0; i < likelyDistribution.length; i++ ){
            runningTotal += likelyDistribution[i];
            if( target <= runningTotal ) {
                return i;
            }
        }

        throw new RuntimeException("Couldn't find a valid possibility. Rolled " + target + " against a total of " + total);
    }
}

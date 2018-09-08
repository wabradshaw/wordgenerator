package com.wabradshaw.ml.wordgenerator;

import com.wabradshaw.ml.wordgenerator.tokenisation.Tokeniser;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.util.stream.Collectors;

public class DataSetGenerator {

    private static final String FILENAME = "/phonemes.txt";
    private static final int IGNORED_LINES = 126;

    private final TokenSet tokenSet;
    private final Tokeniser tokeniser;
    private final String[] contents;

    public DataSetGenerator(TokenSet tokenSet){
        this.tokenSet = tokenSet;
        this.tokeniser = tokenSet.getTokeniser();
        this.contents = loadFile();
    }

    public DataSet getDataSet(int batchNumber, int batchSize, int maxWordLength) {

        String[] batchContents = Arrays.copyOfRange(this.contents,
                                                    batchNumber*batchSize,
                                                    batchNumber*batchSize + batchSize);

        List<List<Integer>> tokens = Arrays.stream(batchContents)
                                           .filter(x -> x != null)
                                           .map(tokeniser::getRelevantWord)
                                           .filter(x -> x != null)
                                           .map(tokeniser::tokenise)
                                           .filter(x -> x.size() <= maxWordLength)
                                           .collect(Collectors.toList());

        List<List<Integer>> mask = tokens.stream()
                                         .map(line -> line.stream()
                                                          .map(c -> c < 0 ? 0 : 1)
                                                          .collect(Collectors.toList())
                                         )
                                         .collect(Collectors.toList());

        return createDataSet(tokens, mask, maxWordLength);
    }

    private String[] loadFile(){
        try {
            File file = new File(this.getClass().getResource(FILENAME).getFile());

            Scanner lineCounter = new Scanner(file);
            int lineCount = 0;
            while(lineCounter.hasNextLine()){
                lineCounter.nextLine();
                lineCount++;
            }

            Scanner scanner = new Scanner(file);

            String[] results = new String[lineCount - IGNORED_LINES];

            int ignoredNumber = 0;
            int lineNumber = 0;
            while(scanner.hasNextLine()) {
                String line = scanner.nextLine();
                if(ignoredNumber >= IGNORED_LINES){
                    results[lineNumber] = line;
                    lineNumber++;
                } else {
                    ignoredNumber++;
                }
            }

            return results;

        } catch(FileNotFoundException e){
            throw new RuntimeException("Phonemes file could not be found.", e);
        }
    }

    private DataSet createDataSet(List<List<Integer>> tokens, List<List<Integer>> mask, int maxWordLength){
        INDArray input = Nd4j.zeros(new int[]{tokens.size(), tokenSet.getLength(), maxWordLength + 1}, 'f');
        INDArray labels = Nd4j.zeros(new int[]{tokens.size(), tokenSet.getLength(), maxWordLength + 1}, 'f');
        INDArray inputMask = Nd4j.zeros(new int[]{tokens.size(), maxWordLength + 1}, 'f');
        INDArray labelsMask = Nd4j.zeros(new int[]{tokens.size(), maxWordLength + 1}, 'f');

        for(int entryId = 0; entryId < tokens.size(); entryId ++){
            List<Integer> entry = tokens.get(entryId);

            input.putScalar(new int[]{entryId, 0, 0}, 1);
            inputMask.putScalar(new int[]{entryId, 0}, 1);

            int contentLength = Math.min(entry.size(), maxWordLength);
            for(int charId = 0; charId < contentLength; charId++){
                int c = entry.get(charId);
                input.putScalar(new int[]{entryId, c, charId + 1}, 1);
                labels.putScalar(new int[]{entryId, c, charId}, 1);
                inputMask.putScalar(new int[]{entryId, charId + 1}, 1);
                labelsMask.putScalar(new int[]{entryId, charId}, 1);
            }

            // Add the end of file character (without masking) once the word is finished
            if(contentLength < maxWordLength){
                input.putScalar(new int[]{entryId, 1, contentLength + 1}, 1);
                labels.putScalar(new int[]{entryId, 1, contentLength}, 1);
                inputMask.putScalar(new int[]{entryId, contentLength + 1}, 1);
                labelsMask.putScalar(new int[]{entryId, contentLength}, 1);
            }

            // Pad the output with masking characters
            for(int charId = entry.size() + 1; charId < maxWordLength; charId++){
                input.putScalar(new int[]{entryId, 1, charId + 1}, 1);
                labels.putScalar(new int[]{entryId, 1, charId}, 1);
                inputMask.putScalar(new int[]{entryId, charId + 1}, 0);
                labelsMask.putScalar(new int[]{entryId, charId}, 0);
            }
            labels.putScalar(new int[]{entryId, 1 ,maxWordLength}, 1);
            labelsMask.putScalar(new int[]{entryId, maxWordLength}, 0);
        }

        return new DataSet(input, labels, inputMask, labelsMask);
    }
}

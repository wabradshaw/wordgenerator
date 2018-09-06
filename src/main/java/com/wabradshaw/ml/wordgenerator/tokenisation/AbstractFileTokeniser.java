package com.wabradshaw.ml.wordgenerator.tokenisation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.*;
import java.util.stream.Collectors;

public abstract class AbstractFileTokeniser implements Tokeniser {

    private static final String FILENAME = "/phonemes.txt";
    private static final int IGNORED_LINES = 126;
    protected static final int MAX_LENGTH = 14;

    protected final String[] possibleTokens;
    protected final Map<String, Integer> tokenToIndexMap;

    public AbstractFileTokeniser(String[] possibleTokens){
        this.possibleTokens = possibleTokens;
        this.tokenToIndexMap = new HashMap<>();

        for(int i = 0; i < this.possibleTokens.length; i++){
            tokenToIndexMap.put(this.possibleTokens[i], i);
        }
    }


    @Override
    public DataSet getTokens() {
        List<String> contents = loadFile();

        List<List<Integer>> tokens = contents.parallelStream()
                                             .map(this::getRelevantWord)
                                             .filter(x -> x != null)
                                             .map(this::toTokens)
                                             .collect(Collectors.toList());
        //TODO REMOVE THIS!
        tokens = tokens.subList(0, 5000);

        List<List<Integer>> mask = tokens.stream()
                                         .map(line -> line.stream()
                                                          .map(c -> c < 0 ? 0 : 1)
                                                          .collect(Collectors.toList())
                                         )
                                         .collect(Collectors.toList());

        return createDataSet(tokens, mask);
    }

    @Override
    public String getToken(int index){
        return this.possibleTokens[index];
    }

    protected abstract String getRelevantWord(String line);

    protected abstract List<Integer> toTokens(String word);

    private List<String> loadFile(){
        try {
            List<String> results = new ArrayList<>();

            File file = new File(this.getClass().getResource(FILENAME).getFile());
            Scanner scanner = new Scanner(file);

            int lineNumber = 0;
            while(scanner.hasNextLine()) {
                String line = scanner.nextLine();
                if(lineNumber >= IGNORED_LINES){
                    results.add(line);
                }
                lineNumber++;
            }

            return results;

        } catch(FileNotFoundException e){
            throw new RuntimeException("Phonemes file could not be found.", e);
        }
    }

    private DataSet createDataSet(List<List<Integer>> tokens, List<List<Integer>> mask){
        INDArray input = Nd4j.zeros(new int[]{tokens.size(), possibleTokens.length, MAX_LENGTH + 1}, 'f');
        INDArray labels = Nd4j.zeros(new int[]{tokens.size(), possibleTokens.length, MAX_LENGTH + 1}, 'f');
        INDArray inputMask = Nd4j.zeros(new int[]{tokens.size(), MAX_LENGTH + 1}, 'f');
        INDArray labelsMask = Nd4j.zeros(new int[]{tokens.size(), MAX_LENGTH + 1}, 'f');

        for(int entryId = 0; entryId < tokens.size(); entryId ++){
            List<Integer> entry = tokens.get(entryId);

            input.putScalar(new int[]{entryId, 0, 0}, 1.0);
            inputMask.putScalar(new int[]{entryId, 0}, 1.0);

            int maxLength = Math.min(entry.size(), MAX_LENGTH);
            for(int charId = 0; charId < maxLength; charId++){
                int c = entry.get(charId);
                input.putScalar(new int[]{entryId, c, charId + 1}, 1.0);
                labels.putScalar(new int[]{entryId, c, charId}, 1.0);
                inputMask.putScalar(new int[]{entryId, charId + 1}, 1.0);
                labelsMask.putScalar(new int[]{entryId, charId}, 1.0);
            }

            // Add the end of file character (without masking) once the word is finished
            if(entry.size() < MAX_LENGTH){
                input.putScalar(new int[]{entryId, 1, entry.size() + 1}, 1.0);
                labels.putScalar(new int[]{entryId, 1, entry.size()}, 1.0);
                inputMask.putScalar(new int[]{entryId, entry.size() + 1}, 1.0);
                labelsMask.putScalar(new int[]{entryId, entry.size()}, 1.0);
            }

            // Pad the output with masking characters
            for(int charId = entry.size() + 1; charId < MAX_LENGTH; charId++){
                input.putScalar(new int[]{entryId, 1, charId + 1}, 1.0);
                labels.putScalar(new int[]{entryId, 1, charId}, 1.0);
                inputMask.putScalar(new int[]{entryId, charId + 1}, 0);
                labelsMask.putScalar(new int[]{entryId, charId}, 0);
            }
            labels.putScalar(new int[]{entryId, 1 ,MAX_LENGTH}, 1.0);
            labelsMask.putScalar(new int[]{entryId, MAX_LENGTH}, 0);
        }

        return new DataSet(input, labels, inputMask, labelsMask);
    }
}

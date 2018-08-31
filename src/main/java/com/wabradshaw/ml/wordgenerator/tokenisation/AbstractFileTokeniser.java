package com.wabradshaw.ml.wordgenerator.tokenisation;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public abstract class AbstractFileTokeniser implements Tokeniser {

    private static final String FILENAME = "phonemes.txt";
    private static final int IGNORED_LINES = 126;
    protected final String[] possibleTokens;

    public AbstractFileTokeniser(String[] possibleTokens){
        this.possibleTokens = possibleTokens;
    }

    @Override
    public INDArray getTokens() {
        List<String> contents = loadFile();
        // Remove bad lines

        // Split according to implementation
        // Tokenise according to implementation

        // Vectorise
        // Convert to INDArray
        return null;
    }

    private List<String> loadFile(){
        try {
            List<String> results = new ArrayList<>();

            File file = new File(this.getClass().getResource("/phonemes.txt").getFile());
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
}

package com.wabradshaw.ml.wordgenerator.tokenisation;

import com.wabradshaw.ml.wordgenerator.TokenSet;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class LexemeTokeniser extends AbstractTokeniser {

    private static final int MAX_LENGTH = 14;

    public LexemeTokeniser(TokenSet tokenSet) {
        super(tokenSet);
    }

    @Override
    public String getRelevantWord(String line) {
        String word = line.split(" ")[0];

        //Ignore words (return null) that contain unknown characters
        for(String letter : word.split("")){
            if (!knownSymbol(letter)) {
                return null;
            }
        }

        return word;
    }

    @Override
    public List<Integer> tokenise(String word) {

        String[] letters = word.split("");
        return Arrays.stream(letters)
              .map(this::toToken)
              .collect(Collectors.toList());
    }
}

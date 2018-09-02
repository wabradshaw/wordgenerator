package com.wabradshaw.ml.wordgenerator.tokenisation;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class LexemeTokeniser extends AbstractFileTokeniser {

    public LexemeTokeniser(String[] possibleTokens) {
        super(possibleTokens);
    }

    @Override
    protected String getRelevantWord(String line) {
        String word = line.split(" ")[0];
        for(String letter : word.split("")){
            if (!this.tokenToIndexMap.containsKey(letter)) {
                return null;
            }
        }
        return word;
    }

    @Override
    protected List<Integer> toTokens(String word) {

        String[] letters = word.split("");
        return Arrays.stream(letters)
              .map(x -> this.tokenToIndexMap.get(x))
              .collect(Collectors.toList());
    }
}

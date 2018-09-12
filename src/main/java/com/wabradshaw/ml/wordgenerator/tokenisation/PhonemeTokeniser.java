package com.wabradshaw.ml.wordgenerator.tokenisation;

import com.wabradshaw.ml.wordgenerator.TokenSet;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class PhonemeTokeniser extends AbstractTokeniser {

    private final boolean splitStresses;

    public PhonemeTokeniser(TokenSet tokenSet, boolean splitStresses) {
        super(tokenSet);
        this.splitStresses = splitStresses;
    }

    @Override
    public String getRelevantWord(String line) {
        String[] split = line.split(" ", 2);

        if(split.length == 2){
            return split[1];
        } else {
            // Line is improperly formed and should be ignored.
            return null;
        }
    }

    @Override
    public List<Integer> tokenise(String word) {
        if(this.splitStresses){
            word = word.replaceAll("(\\d)", " $1");
        }

        String[] symbols = word.split(" ");
        return Arrays.stream(symbols)
                     .filter(x -> !x.isEmpty())
                     .map(this::toToken)
                     .collect(Collectors.toList());
    }

    @Override
    public String toSymbol(int token){
        String base = super.toSymbol(token);
        return " " + base;
    }
}

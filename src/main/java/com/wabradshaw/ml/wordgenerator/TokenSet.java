package com.wabradshaw.ml.wordgenerator;

import com.wabradshaw.ml.wordgenerator.tokenisation.LexemeTokeniser;
import com.wabradshaw.ml.wordgenerator.tokenisation.PhonemeTokeniser;
import com.wabradshaw.ml.wordgenerator.tokenisation.Tokeniser;

import java.util.function.Function;
import java.util.function.Supplier;

public enum TokenSet {
    CHARS_EN_CAPS(x -> new LexemeTokeniser(x),
        "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z"),

    CHARS_EN_CAPS_WITH_COMMON(x -> new LexemeTokeniser(x),
        "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,-,'"),

    PHONEMES_ARPABET_SEPARATE_STRESSES(x -> new PhonemeTokeniser(x, true),
        "AA,AE,AH,AO,AW,AY,B,CH,D,DH,EH,ER,EY,F,G,HH,IH,IY,JH,K,L,M,N,NG,OW,OY,P,R,S,SH,T,TH,UH,UW,V,W,Y,Z,ZH,0,1,2"),

    PHONEMES_ARPABET_STRESSED(x -> new PhonemeTokeniser(x, false),
        "AA,AA0,AA1,AA2,AE,AE0,AE1,AE2,AH,AH0,AH1,AH2,AO,AO0,AO1,AO2,AW,AW0,AW1,AW2,AY,AY0,AY1,AY2,B,CH,D,DH,EH," +
        "EH0,EH1,EH2,ER,ER0,ER1,ER2,EY,EY0,EY1,EY2,F,G,HH,IH,IH0,IH1,IH2,IY,IY0,IY1,IY2,JH,K,L,M,N,NG,OW,OW0,OW1,OW2," +
        "OY,OY0,OY1,OY2,P,R,S,SH,T,TH,UH,UH0,UH1,UH2,UW,UW0,UW1,UW2,V,W,Y,Z,ZH");

    public static final String START_TOKEN = "^";
    public static final String EOF_TOKEN = "$";

    private final String[] tokens;
    private final Function<TokenSet, Tokeniser> getTokeniser;

    TokenSet(Function<TokenSet, Tokeniser> getTokeniser, String tokenString){
        this.getTokeniser = getTokeniser;
        this.tokens = (START_TOKEN + "," + EOF_TOKEN + "," + tokenString).split(",");
    }

    public String[] getTokens(){
        return this.tokens.clone();
    }

    public int getLength(){
        return this.tokens.length;
    }

    /**
     * Builds the {@link Tokeniser} that can split a file into characters for this {@link TokenSet}.
     *
     * @return A {@link Tokeniser}
     */
    public Tokeniser getTokeniser(){
       return this.getTokeniser.apply(this);
    }
}

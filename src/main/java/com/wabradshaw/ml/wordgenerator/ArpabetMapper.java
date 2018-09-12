package com.wabradshaw.ml.wordgenerator;

import java.util.*;
import java.util.stream.Collectors;

public class ArpabetMapper {

    private final Map<String, String> mappings;

    private final Set <String> vowels = new HashSet<>(Arrays.asList("AA,AE,AH,AO,AW,AY,EH,ER,EY,IH,IY,OW,OY,UH,UW".split(",")));
    private final Set <String> irreplaceable = new HashSet<>(Arrays.asList("IY,0,1,2".split(",")));
    private final Set <String> stresses = new HashSet<>(Arrays.asList("1","2"));

    public ArpabetMapper(){
        mappings = new HashMap<>();
        mappings.put("","");
        mappings.put(" ","");
        mappings.put("AA","ɑ");
        mappings.put("AE","æ");
        mappings.put("AH","ʌ");
        mappings.put("AO","ɔ");
        mappings.put("AW","aʊ");
        mappings.put("AY","aɪ");
        mappings.put("B","b");
        mappings.put("CH","t͡ʃ");
        mappings.put("D","d");
        mappings.put("DH","ð");
        mappings.put("EH","ɛ");
        mappings.put("ER","ɝ");
        mappings.put("EY","eɪ");
        mappings.put("F","f");
        mappings.put("G","ɡ");
        mappings.put("HH","h");
        mappings.put("IH","ɪ");
        mappings.put("IY","i");
        mappings.put("JH","d͡ʒ");
        mappings.put("K","k");
        mappings.put("L","l");
        mappings.put("M","m");
        mappings.put("N","n");
        mappings.put("NG","ŋ");
        mappings.put("OW","oʊ");
        mappings.put("OY","ɔɪ");
        mappings.put("P","p");
        mappings.put("R","ɹ");
        mappings.put("S","s");
        mappings.put("SH","ʃ");
        mappings.put("T","t");
        mappings.put("TH","θ");
        mappings.put("UH","ʊ");
        mappings.put("UW","u");
        mappings.put("V","v");
        mappings.put("W","w");
        mappings.put("Y","j");
        mappings.put("Z","z");
        mappings.put("ZH","ʒ");
        mappings.put("0","");
        mappings.put("1","ˈ");
        mappings.put("2","ˌ");
        mappings.put("ə","ə");
    }

    /**
     * Maps a string of arpabet characters into the Alexa dialect of IPA. To make matters more complex, IPA puts the
     * stresses on the
     *
     * @param arpabet The string of arpabet characters, with split stresses (e.g. "IH 0 G Z AE 1 M P AH 0 L")
     * @return        The IPA version of the input string (e.g.
     */
    public String map(String arpabet){
        String[] symbols = arpabet.split(" ");

        List<String> reordered = new ArrayList<>();
        for(int i = 0; i < symbols.length; i ++){
            String symbol = symbols[i];

            // Stress symbols should be added before the first consonant before the symbol
            if(stresses.contains(symbol)){
                int targetIndex = i - 1;
                while(targetIndex > 0 && vowels.contains(symbols[targetIndex])){
                    targetIndex--;
                }
                reordered.add(targetIndex, symbol);

            // Unstressed vowels (0) almost always use ə instead of whatever they started with.
            } else if (symbol.equals("0") && i > 0){
                String previous = reordered.get(reordered.size()-1);
                if(!irreplaceable.contains(previous)) {
                    reordered.set(reordered.size() - 1, "ə");
                }
                // Unstressed symbol doesn't really need to be there, but this makes indexes a lot easier for reordering
                reordered.add(symbol);

            // Everything else can be added
            } else {
                reordered.add(symbol);
            }

        }

        return String.join("", reordered.stream().map(mappings::get).collect(Collectors.toList()));
    }

    public static void main(String[] args){
        ArpabetMapper mapper = new ArpabetMapper();
        System.out.println(mapper.map("F R OW 1 Z AH 0 N"));
    }
}

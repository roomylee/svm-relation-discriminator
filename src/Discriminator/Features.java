package Discriminator;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.util.CoreMap;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

public class Features implements Serializable{
    private List<String> corpus;
    private List<List<String>> tokens;
    private List<List<Pair<Integer, Double>>> vector;
    private int feature_counts;
    String TFIDF_DIR = "model/tfidf.model";

    TfidfVectorizer tfidf_vectorizer;

    Features(){
        tfidf_vectorizer = new TfidfVectorizer();
    }

    public List<List<Pair<Integer, Double>>> getFeatureVector(List<String> corpus, boolean is_training) {
        this.corpus = corpus;
        this.tokens = tokenize(corpus);
        this.vector = new ArrayList<>();
        for(int i=0; i<corpus.size(); i++)
            this.vector.add(new ArrayList<>());
        this.feature_counts = 0;

        if(is_training)
            tfidf_vectorizer = new TfidfVectorizer();

        feature_counts += add_tfidf(is_training);
        feature_counts += add_length(feature_counts);
        feature_counts += add_entity_distance(feature_counts);
        feature_counts += add_pos_between_entity(feature_counts);
        feature_counts += add_relational_word(feature_counts);

        return this.vector;
    }

    private List<List<String>> tokenize(List<String> corpus)
    {
        List<List<String>> tokens = new ArrayList<>();
        for (String sentence : corpus) {
            Reader r = new StringReader(sentence);
            PTBTokenizer<Word> tokenizer = PTBTokenizer.newPTBTokenizer(r);
            List<String> words = new ArrayList<>();

            while (tokenizer.hasNext()) {
                Word w = (Word) tokenizer.next();
                words.add(w.word());
            }
            tokens.add(words);
        }
        return tokens;
    }

    public int add_tfidf(boolean is_training){
        List<List<Pair<Integer, Double>>> tfidf_feature;
        try {
            if(is_training) {
                tfidf_vectorizer.fit(this.tokens);
                tfidf_vectorizer.save_model(TFIDF_DIR);
            } else {
                tfidf_vectorizer = tfidf_vectorizer.load_model(TFIDF_DIR);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        tfidf_feature = tfidf_vectorizer.transform(this.tokens);

        for(int i=0; i<tfidf_feature.size(); i++)
            vector.get(i).addAll(tfidf_feature.get(i));

        return tfidf_vectorizer.vocabulary.size();
    }

    public int add_length(int idx){
        for(int i=0; i<this.tokens.size(); i++) {
            List<String> sentence = tokens.get(i);
            // Suppose that mean = 27, std = 20
            vector.get(i).add(new Pair<>(idx, (double) (sentence.size() - 27)/20));
        }
        return 1;
    }

    public int add_entity_distance(int idx){
        for(int i=0; i<this.tokens.size(); i++)
        {
            int dist = 987654321;
            int b=-987654321, d=-987654321;
            for(int j=0; j<this.tokens.get(i).size(); j++) {
                String word = this.tokens.get(i).get(j);
                if (word.length() > 5 && word.substring(0, 5).equals("BAC00")) {
                    dist = Math.min(dist, j-d);
                    b = j;
                }
                if (word.length() > 5 && word.substring(0, 5).equals("DIS00")) {
                    dist = Math.min(dist, j-b);
                    d = j;
                }
            }

            vector.get(i).add(new Pair<>(idx, (double) (dist-10)/10));
        }
        return 1;
    }

    public int add_pos_between_entity(int idx){
        Properties props = new Properties();
        props.setProperty("annotators","tokenize, ssplit, pos");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        for(int i=0; i<this.tokens.size(); i++) {
            Annotation annotation = new Annotation(corpus.get(i));
            pipeline.annotate(annotation);
            List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);

            List<String> words = new ArrayList<>();
            List<String> posTags = new ArrayList<>();
            List<Integer> bac_idx = new ArrayList<>();
            List<Integer> dis_idx = new ArrayList<>();
            int index = 0;
            for (CoreMap sentence : sentences) {
                for (CoreLabel token : sentence.get(CoreAnnotations.TokensAnnotation.class)) {
                    String word = token.get(CoreAnnotations.TextAnnotation.class);
                    if (word.length() > 5 && word.substring(0, 5).equals("BAC00")) {
                        bac_idx.add(index);
                    } else if (word.length() > 5 && word.substring(0, 5).equals("DIS00")) {
                        dis_idx.add(index);
                    }
                    // this is the POS tag of the token
                    String pos = token.get(CoreAnnotations.PartOfSpeechAnnotation.class);
                    words.add(word);
                    posTags.add(pos);
                    index++;
                }
            }

            int mini = 987654321;
            int start=0, end=0;
            for(int b : bac_idx) {
                for(int d : dis_idx) {
                    if(Math.abs(b-d) < mini){
                        mini = Math.abs(b-d);
                        start = Math.min(b, d);
                        end = Math.max(b, d);
                    }
                }
            }
            // pos tags between entities
            int[] tags = new int[7];
            for (int j = start + 1; j < end; j++) {
                if (posTags.get(j).charAt(0) == 'N')
                    tags[0]++;
                else if (posTags.get(j).charAt(0) == 'V')
                    tags[1]++;
                else if (posTags.get(j).charAt(0) == 'J')
                    tags[2]++;
                else if (posTags.get(j).charAt(0) == 'R')
                    tags[3]++;
                else if (posTags.get(j).charAt(0) == 'I')
                    tags[4]++;
                else if (posTags.get(j).charAt(0) == 'W')
                    tags[5]++;
                else
                    tags[6]++;
            }
            for (int j = 0; j < 7; j++)
                vector.get(i).add(new Pair<>(idx + j, (double) tags[j]));
        }
        return 7;
    }

    public int add_relational_word(int idx){
        String[] relational_words = {
                "caus", "induce", "pathogen", "due", "agent", "lead",
                "respons", "associat", "develop", "isolat", "contribut"};
        for(int i=0; i<this.tokens.size(); i++) {
            List<Integer> bac_idx = new ArrayList<>();
            List<Integer> dis_idx = new ArrayList<>();
            for(int j=0; j<this.tokens.get(i).size(); j++) {
                String word = this.tokens.get(i).get(j);
                if (word.length() > 5 && word.substring(0, 5).equals("BAC00")) {
                    bac_idx.add(j);
                } else if (word.length() > 5 && word.substring(0, 5).equals("DIS00")) {
                    dis_idx.add(j);
                }
            }

            int cnt=0;
            for(int b : bac_idx){
                for(int d : dis_idx){
                    for(int j=Math.min(b, d)+1; j<Math.max(b, d); j++){
                        for(String rel_word : relational_words){
                            if (this.tokens.get(i).get(j).startsWith(rel_word)) {
                                cnt++;
                            }
                        }
                    }
                }
            }
            vector.get(i).add(new Pair<>(idx, (double) cnt/5));
        }

        return 1;
    }
}

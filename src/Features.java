import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.process.PTBTokenizer;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class Features implements Serializable{
    private List<List<String>> tokens;
    private List<List<Pair<Integer, Double>>> vector;
    private int feature_counts;
    String TFIDF_DIR = "model/tfidf.model";

    TfidfVectorizer tfidf_vectorizer;

    Features(){
        tfidf_vectorizer = new TfidfVectorizer();
    }

    public List<List<Pair<Integer, Double>>> getFeatureVector(List<String> corpus, boolean is_training) {
        this.tokens = tokenize(corpus);
        this.vector = new ArrayList<>();
        for(int i=0; i<corpus.size(); i++)
            this.vector.add(new ArrayList<>());
        this.feature_counts = 0;

        if(is_training)
            tfidf_vectorizer = new TfidfVectorizer();

        add_tfidf(is_training);
        add_others();

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

    public void add_tfidf(boolean is_training){
        List<List<Pair<Integer, Double>>> tfidf_feature;
        try {
            if(is_training) {
                tfidf_vectorizer.fit(this.tokens);
                tfidf_vectorizer.save_model(TFIDF_DIR);
            } else {
                tfidf_vectorizer.load_model(TFIDF_DIR);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        tfidf_feature = tfidf_vectorizer.transform(this.tokens);
        feature_counts = tfidf_vectorizer.vocabulary.size();

        for(int i=0; i<tfidf_feature.size(); i++)
            vector.get(i).addAll(tfidf_feature.get(i));
    }

    public void add_others(){
        for(int i=0; i<this.tokens.size(); i++) {
            List<String> words = tokens.get(i);
            int bac = 0, dis = 0;
            for (String w : words) {
                if (w.length() > 5)
                    if (w.substring(0, 5).equals("BAC00"))
                        bac++;
                    else if (w.substring(0, 5).equals("DIS00"))
                        dis++;
            }
            vector.get(i).add(new Pair<>(this.feature_counts, (double) bac * 2));
            vector.get(i).add(new Pair<>(this.feature_counts+1, (double) dis * 2));
        }
    }
}

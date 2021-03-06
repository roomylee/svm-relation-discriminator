package Discriminator;

import java.io.*;
import java.util.*;

public class TfidfVectorizer implements Serializable
{
	Set<String> vocabulary;
	Map<String, Integer> vocab2idx;
	Map<Integer, String> idx2vocab;
	Map<Integer, Integer> termFrequency;
	Map<Integer, Double> inverseDocumentFrequency;

	Set<String> stopWord;

	TfidfVectorizer()
	{
		this(null);
	}

	TfidfVectorizer(Set<String> stopWord)
	{
		vocabulary = new HashSet<>();
		vocab2idx = new HashMap<>();
		idx2vocab = new HashMap<>();
		termFrequency = new HashMap<>();
		inverseDocumentFrequency = new HashMap<>();

		vocabulary.add("<UNK>");
		vocab2idx.put("<UNK>", 0);
		idx2vocab.put(0, "<UNK>");
		termFrequency.put(0, 0);
		inverseDocumentFrequency.put(0, 0.0);
		this.stopWord = stopWord;
	}

	private void build_vocabulary_and_frequency(List<List<String>> tokens)
	{
		for(List<String> words : tokens) {
			Set<Integer> indices = new HashSet<>();
			for (String word : words) {
				int wordIndex;
				// already exist
				if (vocabulary.contains(word)) {
					wordIndex = vocab2idx.get(word);
					termFrequency.put(wordIndex, termFrequency.get(wordIndex) + 1);
				}
				// first came out
				else {
					wordIndex = vocabulary.size();
					vocabulary.add(word);
					vocab2idx.put(word, wordIndex);
					idx2vocab.put(wordIndex, word);
					termFrequency.put(wordIndex, 1);
				}
				indices.add(wordIndex);
			}
			// Calculate document frequency
			for (Integer idx : indices) {
				if (inverseDocumentFrequency.containsKey(idx))
					inverseDocumentFrequency.put(idx, inverseDocumentFrequency.get(idx) + 1);
				else
					inverseDocumentFrequency.put(idx, 1.0);
			}
		}
	}

	private static double to_idf(double df, double n)
	{
		return Math.log(n / (1 + df));
	}

	public void fit(List<List<String>> corpus)
	{
		build_vocabulary_and_frequency(corpus);

		int documentSize = corpus.size();
		for (int i = 0; i < vocabulary.size(); i++)
		{
			inverseDocumentFrequency.put(i, to_idf(inverseDocumentFrequency.get(i), documentSize));
		}

		System.out.println("Corpus Size = " + documentSize);
		System.out.println("Vocabulary Size = " + vocabulary.size());
	}

	public List<List<Pair<Integer, Double>>> transform(List<List<String>> tokens) {
		List<List<Pair<Integer, Double>>> tfidf_vector = new ArrayList<>();

		for (List<String> words : tokens) {
			List<Integer> indexedWords = new ArrayList<>();
			for (String word : words) {
				if (vocab2idx.containsKey(word)) {
					int wordIndex = vocab2idx.get(word);
					indexedWords.add(wordIndex);
				}
			}

			// calc term frequency
			List<Double> termFrequency = new ArrayList<>();
			double l2_norm = 0;
			for (int i : indexedWords) {
				double cnt = 0;
				for (int j : indexedWords)
					if (i == j)
						cnt++;
				termFrequency.add(cnt);
				l2_norm += cnt * cnt;
			}

			// normalize term frequency
			l2_norm = Math.sqrt(l2_norm);
			for (int i = 0; i < termFrequency.size(); i++) {
				termFrequency.set(i, termFrequency.get(i) / l2_norm);
			}

			// calc tfidf
			List<Pair<Integer, Double>> tfidfs = new ArrayList<>();
			for (int i = 0; i < termFrequency.size(); i++) {
				int wordIndex = indexedWords.get(i);
				double tfidf = termFrequency.get(i) * inverseDocumentFrequency.get(wordIndex);
				tfidfs.add(new Pair<>(wordIndex, tfidf));
			}

			tfidf_vector.add(tfidfs);
		}

		return tfidf_vector;
	}

	public void save_model(String dir) throws IOException
	{
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(dir));
		oos.writeObject(this);
		oos.close();
	}

	public TfidfVectorizer load_model(String dir) throws IOException
	{
		TfidfVectorizer model = null;
		try {
			ObjectInputStream ois = new ObjectInputStream(new FileInputStream(dir));
			model = (TfidfVectorizer) ois.readObject();
			ois.close();
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}
		return model;
	}
}

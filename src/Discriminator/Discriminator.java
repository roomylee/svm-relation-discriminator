package Discriminator;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Discriminator {
	Features features;
	public SVM svmModel;

	public Discriminator(){
		features = new Features();
		svmModel = new SVM();
	}

	public void train(List<String> corpus, List<Integer> label)
	{
		List<List<Pair<Integer, Double>>> featuresVector = features.getFeatureVector(corpus, true);;
		svmModel.train(featuresVector, label);
	}

	public List<Double> predict(List<String> corpus)
	{
		List<List<Pair<Integer, Double>>> featuresVector = features.getFeatureVector(corpus, false);
		List<Double> pred = svmModel.predict(featuresVector);
		return pred;
	}

	public void cross_validation(List<String> corpus, List<Integer> label, int fold) {
		List<List<Pair<Integer, Double>>> featuresVector = features.getFeatureVector(corpus, true);
		svmModel.do_cross_validation(featuresVector, label, fold);
	}

	public void save_svm(String dir) throws IOException {
		svmModel.save_model(dir);
	}
	public void load_svm(String dir) throws IOException {
		svmModel.model = svmModel.load_model(dir);
	}

	public static void main(String args[]) {

		// Train Phase
		List<String> trainCorpus = new ArrayList<>();
		List<Integer> trainLabel = new ArrayList<>();

		String trainDir = "data/TrainSet.txt";

		FileReader fr;
		BufferedReader br;
		try {
			fr = new FileReader(trainDir);
			br = new BufferedReader(fr);

			String curLine;
			while ((curLine = br.readLine()) != null) {
				String x = curLine.split("\t")[0];
				trainCorpus.add(x);

				String y = curLine.split("\t")[1];
				if (y.equals("None")){
					trainLabel.add(-1);
				}
				else {
					trainLabel.add(+1);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		// Train SVM
		String svm_dir = "model/svm.model";
		Discriminator d = new Discriminator();
		d.train(trainCorpus, trainLabel);
		try {
			d.save_svm(svm_dir);
			d.load_svm(svm_dir);
		} catch (IOException e) {
			e.printStackTrace();
		}


		// Test(Evaluate) Phase
		List<String> testCorpus = new ArrayList<>();
		List<Integer> testLabel = new ArrayList<>();
		String testDir = "data/TestSet.txt";

		try {
			fr = new FileReader(testDir);
			br = new BufferedReader(fr);

			String curLine;
			while ((curLine = br.readLine()) != null) {
				String x = curLine.split("\t")[0];
				testCorpus.add(x);

				String y = curLine.split("\t")[1];
				if (y.equals("None")){
					testLabel.add(-1);
				}
				else {
					testLabel.add(+1);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		List<Double> pred = d.predict(testCorpus);
		int tp=0, tn=0, fp=0, fn=0;
		double threshold=0.5;
		for(int i=0; i<pred.size(); i++)
		{
			if(pred.get(i) > threshold) {
				if(testLabel.get(i) == 1)
					++tp;
				else
					++fp;
			}
			else {
				if(testLabel.get(i) == -1)
					++tn;
				else
					++fn;
			}
		}
		System.out.println("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
		System.out.println("Precision = "+(double)(tp)/(tp+fp)*100+ "% ("+(tp)+"/"+(tp+fp)+")");
		System.out.println("Recall = "+(double)(tp)/(tp+fn)*100+ "% ("+(tp)+"/"+(tp+fn)+")");
		System.out.println("Accuracy = "+(double)(tp+tn)/(tp+tn+fp+fn)*100+ "% ("+(tp+tn)+"/"+(tp+tn+fp+fn)+")");
		System.out.println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");

		d.svmModel.param.probability = 0;
		d.cross_validation(trainCorpus, trainLabel, 10);
	}
}

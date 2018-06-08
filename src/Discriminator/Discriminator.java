package Discriminator;

import java.io.BufferedReader;
import java.io.File;
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

	public static List<String> listFilesForDirectory(final File dir){
		List<String> fileList = new ArrayList<>();
		for(final File fileEntry : dir.listFiles()){
			if(fileEntry.isDirectory()){
				fileList.addAll(listFilesForDirectory(fileEntry));
			} else{
				fileList.add(fileEntry.getName());
			}
		}
		return fileList;
	}

	public static void main(String args[]) {
		List<String> corpus = new ArrayList<>();
		List<Integer> label = new ArrayList<>();

		final File dir = new File("data");
		List<String> fileList = listFilesForDirectory(dir);

		for(String fileName : fileList) {
			FileReader fr;
			BufferedReader br;
			try {
				fr = new FileReader("data" + "/" + fileName);
				br = new BufferedReader(fr);

				String curLine;
				while ((curLine = br.readLine()) != null) {
					String x = curLine.split("\t")[0];
					corpus.add(x);

					String y = curLine.split("\t")[1];
					if (y.equals("None")){
						label.add(-1);
					}
					else {
						label.add(+1);
					}
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}


		String svm_dir = "model/svm.model";
		Discriminator d = new Discriminator();
		d.train(corpus, label);
		try {
			d.save_svm(svm_dir);
		} catch (IOException e) {
			e.printStackTrace();
		}

		try {
			d.load_svm(svm_dir);
		} catch (IOException e) {
			e.printStackTrace();
		}

		List<Double> pred = d.predict(corpus);
		int tp=0, tn=0, fp=0, fn=0;
		double threshold=0.5;
		for(int i=0; i<pred.size(); i++)
		{
			if(pred.get(i) > threshold) {
				if(label.get(i) == 1)
					++tp;
				else
					++fp;
			}
			else {
				if(label.get(i) == -1)
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
		d.cross_validation(corpus, label, 10);
	}
}

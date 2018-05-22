import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Discriminator {
	public TfidfVectorizer tfidfModel;
	public SVM svmModel;

	private Discriminator(){
		tfidfModel = new TfidfVectorizer();
		svmModel = new SVM();
	}

	public Discriminator(String tfidfDir, String svmDir) {
		// 파라미터의 경로로부터 객체 불러오기
		tfidfModel = new TfidfVectorizer();
		try {
			tfidfModel = tfidfModel.load_model(tfidfDir);
		} catch (IOException e) {
			System.out.printf("\"%s\" is not found.\n", tfidfDir);
		}

		svmModel = new SVM();
		try{
			svmModel.model = svmModel.load_model(svmDir);
		} catch (IOException e) {
			System.out.printf("\"%s\" is not found.\n", svmDir);
		}
	}

	public void train(List<String> corpus, List<Integer> label)
	{
		tfidfModel.fit(corpus);
		List<List<Pair<Integer, Double>>> tfidf_vec = tfidfModel.transform(corpus);
		svmModel.train(tfidf_vec, label);
	}

	public List<Integer> predict(List<String> corpus)
	{
		List<List<Pair<Integer, Double>>> x = tfidfModel.transform(corpus);
		List<Double> pred_prob = svmModel.predict(x);
		List<Integer> pred = new ArrayList<>();

		double threshold = 0.5;
		for(int i=0; i<pred_prob.size(); i++)
		{
			if(pred_prob.get(i) < threshold){
				pred.add(0);
			}
			else{
				pred.add(1);
			}
		}

		return pred;
	}

	public void cross_validation(List<String> corpus, List<Integer> label, int fold) {
		tfidfModel.fit(corpus);
		List<List<Pair<Integer, Double>>> tfidf_vec = tfidfModel.transform(corpus);
		svmModel.do_cross_validation(tfidf_vec, label, fold);
	}

	public void save_tfidf(String dir) throws IOException {
		tfidfModel.save_model(dir);
	}
	public void save_svm(String dir) throws IOException {
		svmModel.save_model(dir);
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
			FileReader fr = null;
			BufferedReader br = null;
			try {
				fr = new FileReader("data" + "/" + fileName);
				br = new BufferedReader(fr);

				String curLine;
				while ((curLine = br.readLine()) != null) {
					corpus.add(curLine.split("\t")[0]);
					if (curLine.split("\t")[1].equals("None")){
						label.add(0);
					}
					else {
						label.add(1);
					}
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		}


		String tfidf_dir = "model/tfidf.model";
		String svm_dir = "model/svm.model";
		Discriminator d = new Discriminator();
		d.train(corpus, label);
		try {
			d.save_tfidf(tfidf_dir);
			d.save_svm(svm_dir);
		} catch (IOException e) {
			e.printStackTrace();
		}
		List<Integer> pred = d.predict(corpus);
		int correct=0;
		int total=0;
		for(int i=0; i<pred.size(); i++)
		{
			if(pred.get(i).equals(label.get(i))){
				++correct;
			}
			++total;
		}
		System.out.print("Accuracy = "+(double)correct/total*100+ "% ("+correct+"/"+total+") (classification)\n\n");


		System.out.println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");


		d.cross_validation(corpus, label, 10);
	}
}

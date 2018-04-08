import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

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
		List<List<Map.Entry<Integer, Double>>> tfidf_vec = tfidfModel.transform(corpus);
		svmModel.train(tfidf_vec, label);
	}

	public List<Integer> predict(List<String> corpus)
	{
		List<List<Map.Entry<Integer, Double>>> x = tfidfModel.transform(corpus);
		List<Integer> pred = svmModel.predict(x);

		return pred;
	}

	public List<Integer> predict(List<String> corpus, List<Integer> label)
	{
		List<List<Map.Entry<Integer, Double>>> x = tfidfModel.transform(corpus);
		List<Integer> pred = svmModel.predict(x, label);

		return pred;
	}

	public void cross_validation(List<String> corpus, List<Integer> label, int fold) {
		tfidfModel.fit(corpus);
		List<List<Map.Entry<Integer, Double>>> tfidf_vec = tfidfModel.transform(corpus);
		svmModel.cross_validation(tfidf_vec, label, fold);
	}

	public void save_tfidf(String dir) throws IOException {
		tfidfModel.save_model(dir);
	}
	public void save_svm(String dir) throws IOException {
		svmModel.save_model(dir);
	}

	public static List<String> listFilesForDirectory(final File dir){
		List<String> fileList = new ArrayList<String>();
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
					if (curLine.split("\t")[1].equals("None")
							|| curLine.split("\t")[1].equals("0")){
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
		List<Integer> pred = d.predict(corpus, label);

		d.cross_validation(corpus, label, 10);

//		for(int i : d.predict(corpus)) {
//			System.out.println(i);
//		}
	}
}

import libsvm.*;
import java.io.*;
import java.util.*;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class SVM {
	svm_parameter param;
	svm_problem prob;
	svm_model model;
	String error_msg;

	public void init_parameter()
	{
		param = new svm_parameter();
		// default values
		param.svm_type = svm_parameter.NU_SVC;
		param.kernel_type = svm_parameter.RBF;
		param.degree = 3;
		param.gamma = 0;	// 1/num_features
		param.coef0 = 0;
		param.nu = 0.5;
		param.cache_size = 100;
		param.C = 1;
		param.eps = 1e-3;
		param.p = 0.1;
		param.shrinking = 1;
		param.probability = 0;
		param.nr_weight = 0;
		param.weight_label = new int[0];
		param.weight = new double[0];
	}

	public void data_processing(List<List<Map.Entry<Integer, Double>>> data, List<Integer> label)
	{
		Vector<Double> vy = new Vector<Double>();
		Vector<svm_node[]> vx = new Vector<svm_node[]>();
		int max_index = 0;

		for (int i=0; i<data.size(); i++)
		{
			List<Map.Entry<Integer, Double>> x = data.get(i);
			double y = label.get(i);

			vy.addElement(y);

			int m = x.size();
			svm_node[] xx = new svm_node[m];
			for(int j=0; j<m; j++)
			{
				xx[j] = new svm_node();
				xx[j].index = x.get(j).getKey();
				xx[j].value = x.get(j).getValue();
			}
			if(m>0) max_index = Math.max(max_index, xx[m-1].index);
			vx.addElement(xx);
		}

		prob = new svm_problem();
		prob.l = vy.size();
		prob.x = new svm_node[prob.l][];
		for(int i=0;i<prob.l;i++)
			prob.x[i] = vx.elementAt(i);
		prob.y = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			prob.y[i] = vy.elementAt(i);

		if(param.gamma == 0 && max_index > 0)
			param.gamma = 1.0/max_index;

		if(param.kernel_type == svm_parameter.PRECOMPUTED)
			for(int i=0;i<prob.l;i++)
			{
				if (prob.x[i][0].index != 0)
				{
					System.err.print("Wrong kernel matrix: first column must be 0:sample_serial_number\n");
					System.exit(1);
				}
				if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
				{
					System.err.print("Wrong input format: sample_serial_number out of range\n");
					System.exit(1);
				}
			}
	}

	public void train(List<List<Map.Entry<Integer, Double>>> data, List<Integer> label)
	{
		init_parameter();
		data_processing(data, label);

		error_msg = svm.svm_check_parameter(prob,param);

		if(error_msg != null)
		{
			System.err.print("ERROR: "+error_msg+"\n");
			System.exit(1);
		}

		model = svm.svm_train(prob,param);
	}

	public List<Integer> predict(List<List<Map.Entry<Integer, Double>>> data, List<Integer> label)
	{
		List<Integer> output = new ArrayList<>();
		int correct = 0;
		int total = 0;

		for (int i=0; i<data.size(); i++)
		{
			List<Map.Entry<Integer, Double>> x = data.get(i);
			double y = label.get(i);

			int m = x.size();
			svm_node[] xx = new svm_node[m];
			for (int j=0; j<m; j++)
			{
				xx[j] = new svm_node();
				xx[j].index = x.get(j).getKey();
				xx[j].value = x.get(j).getValue();
			}

			double pred = svm.svm_predict(model,xx);
			output.add((int) pred);

			if(pred == y)
				++correct;
			++total;
		}

		System.out.print("Accuracy = "+(double)correct/total*100+ "% ("+correct+"/"+total+") (classification)\n\n");

		return output;
	}


	public List<Integer> predict(List<List<Map.Entry<Integer, Double>>> data)
	{
		List<Integer> output = new ArrayList<>();
		for (int i=0; i<data.size(); i++)
		{
			List<Map.Entry<Integer, Double>> x = data.get(i);

			int m = x.size();
			svm_node[] xx = new svm_node[m];
			for (int j=0; j<m; j++)
			{
				xx[j] = new svm_node();
				xx[j].index = x.get(j).getKey();
				xx[j].value = x.get(j).getValue();
			}

			double pred = svm.svm_predict(model,xx);
			output.add((int) pred);
		}
		return output;
	}

	public void cross_validation(List<List<Map.Entry<Integer, Double>>> data, List<Integer> label, int fold){
		long random_seed = 10;
		List<Integer> index = IntStream.range(1,prob.l).boxed().collect(Collectors.toList());
		Collections.shuffle(index, new Random(random_seed));

		List<List<Map.Entry<Integer, Double>>> train_x = new ArrayList<>();
		List<List<Map.Entry<Integer, Double>>> dev_x = new ArrayList<>();
		List<Integer> train_y = new ArrayList<>();
		List<Integer> dev_y = new ArrayList<>();
		for (int i=0; i<index.size(); i++)
		{
			if (i < (double) index.size()/fold) {
				dev_x.add(data.get(index.get(i)));
				dev_y.add(label.get(index.get(i)));
			} else {
				train_x.add(data.get(index.get(i)));
				train_y.add(label.get(index.get(i)));
			}
		}

		train(train_x, train_y);

		System.out.print(fold + "-Fold CV ");
		predict(dev_x, dev_y);
	}

	public void save_model(String dir) throws IOException {
		svm.svm_save_model(dir, this.model);
	}

	public svm_model load_model(String dir) throws IOException {
		return svm.svm_load_model(dir);
	}

	public static void main(String args[]) throws IOException {
		TfidfVectorizer tfidf = new TfidfVectorizer();
		List<String> corpus = new ArrayList<>();
		corpus.add("How am i happy.");
		corpus.add("Are you happy?");
		corpus.add("we are bad.");
		corpus.add("Why so serious?");

		tfidf.fit(corpus);
		List<List<Map.Entry<Integer, Double>>> d = tfidf.transform(corpus);
		List<Integer> l = new ArrayList<>();
		l.add(1);
		l.add(0);
		l.add(0);
		l.add(1);

		SVM svm = new SVM();
		svm.train(d, l);
		svm.predict(d, l);
		for (int p : svm.predict(d))
		{
			System.out.println(p);
		}
	}
}

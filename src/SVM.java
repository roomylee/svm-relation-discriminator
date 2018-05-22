import libsvm.*;
import java.io.*;
import java.util.*;

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

	public void data_processing(List<List<Pair<Integer, Double>>> data, List<Integer> label)
	{
		Vector<Double> vy = new Vector<>();
		Vector<svm_node[]> vx = new Vector<>();
		int max_index = 0;

		for (int i=0; i<data.size(); i++)
		{
			List<Pair<Integer, Double>> x = data.get(i);
			double y = label.get(i);

			vy.addElement(y);

			int m = x.size();
			svm_node[] xx = new svm_node[m];
			for(int j=0; j<m; j++)
			{
				xx[j] = new svm_node();
				xx[j].index = x.get(j).getFirst();
				xx[j].value = x.get(j).getSecond();
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

	public void train(List<List<Pair<Integer, Double>>> data, List<Integer> label)
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

	public List<Double> predict(List<List<Pair<Integer, Double>>> data)
	{
		List<Double> output = new ArrayList<>();
		for (int i=0; i<data.size(); i++)
		{
			List<Pair<Integer, Double>> x = data.get(i);

			int m = x.size();
			svm_node[] xx = new svm_node[m];
			for (int j=0; j<m; j++)
			{
				xx[j] = new svm_node();
				xx[j].index = x.get(j).getFirst();
				xx[j].value = x.get(j).getSecond();
			}

			double pred = svm.svm_predict_probability(model, xx, new double[model.nr_class]);
			output.add(pred);
		}
		return output;
	}

	public void do_cross_validation(List<List<Pair<Integer, Double>>> data, List<Integer> label, int fold) {
		init_parameter();
		data_processing(data, label);

		double[] target = new double[prob.l];
		svm.svm_cross_validation(prob, param, fold, target);

		int total_correct = 0;
		for(int i=0;i<prob.l;i++) {
			if (target[i] == prob.y[i])
				++total_correct;
		}
		System.out.print("Cross Validation Accuracy = "+100.0*total_correct/prob.l+"%\n");
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
		List<List<Pair<Integer, Double>>> d = tfidf.transform(corpus);
		List<Integer> l = new ArrayList<>();
		l.add(1);
		l.add(0);
		l.add(0);
		l.add(1);

		SVM svm = new SVM();
		svm.train(d, l);
		svm.predict(d);
	}
}

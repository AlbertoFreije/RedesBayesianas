package es.uniovi.ssii.rb;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.openmarkov.core.exception.IncompatibleEvidenceException;
import org.openmarkov.core.exception.NotEvaluableNetworkException;
import org.openmarkov.core.model.network.EvidenceCase;
import org.openmarkov.core.model.network.Finding;
import org.openmarkov.core.model.network.ProbNet;
import org.openmarkov.core.model.network.Variable;
import org.openmarkov.core.model.network.potential.TablePotential;
import org.openmarkov.gui.dialog.io.NetsIO;
import org.openmarkov.inference.huginPropagation.HuginPropagation;
import org.openmarkov.core.model.network.State;

public class UO257351 {
	
	private ProbNet probNet;
	private Long seed;
	private Random rnd;

	public UO257351(String fileName) throws Exception {
		probNet = NetsIO.openNetworkFile("src/main/resources/networks/"+fileName).getProbNet(); //cargar red bayesiana
		seed = null;
		rnd = new Random();
	}

	public ProbNet getProbNet() {
		return probNet;
	}

	public void setProbNet(ProbNet probNet) {
		this.probNet = probNet;
	}

	public Long getSeed() {
		return seed;
	}

	public void setSeed(Long seed) {
		this.seed = seed;
		rnd.setSeed(seed);
	}
	
	public HuginPropagation HuginInference(List<Variable> variablesOfInterest, EvidenceCase evidence) {

		HuginPropagation propagation = null; //creamos objeto de la clase 
		try {
			propagation = new HuginPropagation(probNet);
		} catch (NotEvaluableNetworkException e) {
			e.printStackTrace();
		}
		propagation.setVariablesOfInterest(variablesOfInterest);

		propagation.setPostResolutionEvidence(evidence);

		System.out.print("Variable elimination\n");
		long startTime = System.nanoTime();
		try {
			Map<Variable, TablePotential> posteriorProbabilities = propagation.getPosteriorValues();
			printProbabilities(evidence, variablesOfInterest, posteriorProbabilities);

		} catch (IncompatibleEvidenceException e) {
			e.printStackTrace();
		} catch (OutOfMemoryError e) {
			e.printStackTrace();
		}
		long endTime = System.nanoTime();

		printTime(endTime - startTime);

		return propagation;
	}
	
	public static void printProbabilities(EvidenceCase evidence, List<Variable> variablesOfInterest,
			Map<Variable, TablePotential> posteriorProbabilities) {

		String evidenceString = "";
		for (Finding finding : evidence.getFindings()) {
			evidenceString += " " + finding.getVariable() + "=" + finding.getState();
		}

		for (Variable variable : variablesOfInterest) {
			TablePotential posteriorProbabilitiesPotential = posteriorProbabilities.get(variable);
			System.out.format("P( %s=%s | %s) = %.5f\n", variable, variable.getStates()[0].getName(), evidenceString,
					posteriorProbabilitiesPotential.values[0]);
		}
	}
	
	public static void printTime(long nanoseconds) {
		System.out.format("Total time: %.2f miliseconds\n", nanoseconds / 1000000.0);
	}
	
	public static void main(String[] args) throws Exception {

		UO257351 obj = new UO257351("asia.pgmx"); //creamos el objeto pasandole la red 
		
		Variable hasLungCancer = obj.getProbNet().getVariable("Has lung cancer");
		List<Variable> variables = Arrays.asList(hasLungCancer);
		
		EvidenceCase evidencia = new EvidenceCase();
		evidencia.addFinding(obj.getProbNet(), "Has tuberculosis", "yes");
		evidencia.addFinding(obj.getProbNet(), "Has bronchitis", "yes");
		
		HuginPropagation propagacion = obj.HuginInference(variables, evidencia);
		Map<Variable, TablePotential> valores = propagacion.getPosteriorValues();
		TablePotential valor = valores.get(hasLungCancer);
		
		int i = 0;
		for (State estado: hasLungCancer.getStates()) {
			System.out.println("Estado: "+estado  + " Valor:" +valor.getValues()[i]);
			i++;
		}

	}

}

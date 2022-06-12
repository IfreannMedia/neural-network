using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ANN : MonoBehaviour
{

    public int numInputs;
    public int numOutputs;
    public int numHidden;
    public int numNPerHidden;
    public double alpha;
    List<Layer> layers = new List<Layer>();

    public ANN(int nI, int nO, int nH, int nPH, double a)
    {
        numInputs = nI;
        numOutputs = nO;
        numHidden = nH;
        numNPerHidden = nPH;
        alpha = a;
        
        if(numHidden > 0)
        {
            layers.Add(new Layer(numNPerHidden, numInputs));

            for (int i = 0; i < numHidden-1; i++)
            {
                layers.Add(new Layer(numNPerHidden, numNPerHidden));
            }

            layers.Add(new Layer(numOutputs, numNPerHidden));
        } else
        {
            layers.Add(new Layer(numOutputs, numInputs));
        }
    }

    // Method to train neural network
    public List<double> Go(List<double> inputValues, List<double> desiredOutput)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();

        if(inputValues.Count != numInputs)
        {
            Debug.Log("ERROR: Number of Inputs must be " + numInputs);
            return outputs;
        }

        inputs = new List<double>(inputValues);

        // loop through layers
        for (int i = 0; i < numHidden + 1; i++)
        {
            if(1 > 0)
            {
                // if not on first layer, the inputs become the outputs of the previous layer
                inputs = new List<double>(outputs);
            }
            outputs.Clear();

            // loop through the neurons
            for (int j = 0; j < layers[i].numNeurons; j++)
            {
                // for each neuron, calculate it's weight multiplied by its input
                // for each of its inputs and each of its weights
                // store that as N
                double N = 0;
                layers[i].neurons[j].inputs.Clear();

                // loop through each neurons input
                for (int k = 0; k < layers[i].neurons[j].numInputs; k++)
                {
                    // add the output of previous layer as the input to this neuron
                    layers[i].neurons[j].inputs.Add(inputs[k]);
                    // adjust N value, add to the weight * input
                    // essentially the dot product of the perceptron
                    N += layers[i].neurons[j].weights[k] * inputs[k];
                }
                // adjust the N value based on bias
                N -= layers[i].neurons[j].bias;
                // set this neurons output to the result of running N through our activation function
                layers[i].neurons[j].output = ActivationFunction(N);
                outputs.Add(layers[i].neurons[j].output);
            }

        }

        UpdateWeights(outputs, desiredOutput);

        return outputs;
    }

}

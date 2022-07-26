using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ANN
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

        if (numHidden > 0)
        {
            layers.Add(new Layer(numNPerHidden, numInputs));

            for (int i = 0; i < numHidden - 1; i++)
            {
                layers.Add(new Layer(numNPerHidden, numNPerHidden));
            }

            layers.Add(new Layer(numOutputs, numNPerHidden));
        }
        else
        {
            layers.Add(new Layer(numOutputs, numInputs));
        }
    }

    // Method to train neural network
    public List<double> Go(List<double> inputValues, List<double> desiredOutput)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();

        if (inputValues.Count != numInputs)
        {
            Debug.Log("ERROR: Number of Inputs must be " + numInputs);
            return outputs;
        }

        inputs = new List<double>(inputValues);
        // loop through layers
        for (int i = 0; i < numHidden + 1; i++)
        {
            if (i > 0)
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
                if (i == numHidden)
                    layers[i].neurons[j].output = ActivationFunctionOutputLayer(N);
                else
                    layers[i].neurons[j].output = ActivationFunction(N);
                outputs.Add(layers[i].neurons[j].output);
            }

        }

        UpdateWeights(outputs, desiredOutput);

        return outputs;
    }

    private void UpdateWeights(List<double> outputs, List<double> desiredOutput)
    {
        double error;
        // i is looping through the layers
        // we iterate backwards because we are taking the error and back propagating it through the network
        for (int i = numHidden; i >= 0; i--)
        {
            // j is looping through the neurons of that layer
            for (int j = 0; j < layers[i].numNeurons; j++)
            {
                // if we are at the end or output layerr
                if (i == numHidden)
                {
                    error = desiredOutput[j] - outputs[j];
                    layers[i].neurons[j].errorGradient = outputs[j] * (1 - outputs[j]) * error;
                    // errorGradient calculated with delta rule: en.wikipedia.org/wiki/Delta_rule
                    // this error gradient describes how responsible this neuron is for the error in percentage
                }
                else
                {
                    layers[i].neurons[j].errorGradient = layers[i].neurons[j].output * (1 - layers[i].neurons[j].output);
                    double errorGradSum = 0;
                    // loop through the neurons of the next layer (which is technically the previous layer in our method)
                    for (int p = 0; p < layers[i + 1].numNeurons; p++)
                    {
                        // add previous errorgradient by wieght
                        errorGradSum += layers[i + 1].neurons[p].errorGradient * layers[i + 1].neurons[p].weights[j];
                    }
                    // multiply current errorgradient by the errorgradsum
                    layers[i].neurons[j].errorGradient *= errorGradSum;
                }

                // k is looping through the inputs of that neuron
                for (int k = 0; k < layers[i].neurons[j].numInputs; k++)
                {
                    // if it's the output layer
                    if (i == numHidden)
                    {
                        error = desiredOutput[j] - outputs[j];
                        // update the wight of the neuron with the alpha (which is the learning rate) by input by error
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * error;
                    }
                    else
                    {
                        // update the wight of the neuron with the alpha (which is the learning rate) by input by the neuron error gradient
                        layers[i].neurons[j].weights[k] += alpha * layers[i].neurons[j].inputs[k] * layers[i].neurons[j].errorGradient;
                    }
                }
                layers[i].neurons[j].bias += alpha * -1 * layers[i].neurons[j].errorGradient;
            }
        }
    }

    // also see en.wikipedia.org/wiki/Activation_function
    private double ActivationFunction(double value)
    {
        // can swap between Step and Sigmmoid
        //return Sigmoid(value);
        //return Step(value);
        //return TanH(value);
        return Relu(value);

    }

    private double ActivationFunctionOutputLayer(double value)
    {
        // can swap between Step and Sigmmoid
        //return Sigmoid(value);
        //return Step(value);
        //return TanH(value);
        return Sigmoid(value);

    }

    // aka binary step
    private double Step(double value)
    {
        if (value < 0) return 0;
        else return 1;
    }

    // aka logistic softstep
    private double Sigmoid(double value)
    {
        // get the exponential of value
        double k = (double)System.Math.Exp(value);
        // return exp over 1 + exp
        return k / (1.0f + k);
    }

    private double TanH(double value)
    {
        return (2 * (Sigmoid(2 * value)) - 1);
    }

    private double Relu(double value)
    {
        if (value > 0) return value;
        else return 0;
    }

    private double LeakyRelu(double value)
    {
        if (value < 0) return 0.01 * value;
        else return value;
    }

}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Neuron : MonoBehaviour
{

    public int numInputs;
    public double bias;
    public double output;
    public double errorGradient;
    public List<double> weights = new List<double>();
    public List<double> inputs = new List<double>();

    public Neuron(int nInputs)
    {
        bias = UnityEngine.Random.Range(-1f, 1f);
        numInputs = nInputs;
        for (int i = 0; i < nInputs; i++)
        {

        }
    }

}

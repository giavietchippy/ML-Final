# ML-Final
This is for the final machine learning project

# Q2a - CNN (CIFAR-10)

```bash
pip install -r requirements.txt
python q2_cnn.py --epochs 40            # 4 configs -> results/results.json + figures
python q2_random_search.py --trials 12  # extension -> results/random_search.json
```
Use a GPU (Google Colab is fine); on CPU expect a long run. All seeds are fixed (SEED = 42).
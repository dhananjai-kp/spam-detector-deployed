# Importing essential libraries
from flask import Flask, render_template, request
import pickle


# filename = 'model.h5'
# classifier = load_model("model.h5")
filename = 'xg_model.pkl'
classifier = pickle.load(open(filename, 'rb'))
token = pickle.load(open('token.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = token.transform(data).toarray()
    	my_prediction = classifier.predict(vect)
    	return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
#	app.debug = True
#	app.run(host = '0.0.0.0.', port = 5000)

from flask import Flask, render_template, request
import neuralnet.predict as pr
from console_logging.console import Console
console = Console()
app=Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():

    # get form variables and type them
    gpa = float(request.form["gpa"])
    score = int(request.form["test_score"])
    console.info("Chancing GPA: %d, SAT: %d"%(gpa,score))
    predictions=[]

    #TODO: implement test type. This is a stub.
    if score<=36:
        predictions=pr.predict(gpa,score,"ACT")
    elif score<=1600:
        predictions=pr.predict(gpa,score,"SAT1600")
    else:
        predictions = pr.predict(gpa,score,"SAT2400")
    ##

    if predictions[0]==1:
        return "Admission is likely."
    else:
        if predictions[0]==0:
            return "Admission is unlikely."
        return "Something went wrong."

@app.route('/')
def home():
    return render_template('website.html', college={'name':'CMU','accuracy':'76.4045'})

if __name__ == '__main__':
    app.debug=True
    app.run(host='0.0.0.0')
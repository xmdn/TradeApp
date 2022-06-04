from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import datetime
from flask_marshmallow import Marshmallow
from matplotlib.pyplot import title

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:''@localhost/tradedb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
ma = Marshmallow(app)

class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100))
    body = db.Column(db.Text())
    date = db.Column(db.DateTime, default = datetime.datetime.now)

    def __init__(self, title, body):
        self.title = title
        self.body = body

class UsersSchema(ma.Schema):
    class Meta:
        fields = ('id', 'title', 'body', 'date')

users_schema = UsersSchema()
musers_schema = UsersSchema(many=True)



@app.route('/get', methods = ['GET'])
def get_content():
    all_users = Users.query.all()
    results = musers_schema.dump(all_users)
    return jsonify(results)

@app.route('/get/<id>/', methods = ['GET'])
def post_details(id):
    users = Users.query.get(id)
    return users_schema.jsonify(users)


@app.route('/add', methods = ['POST'])
def add_content():
    title = request.json['title']
    body = request.json['body']

    contents = Users(title, body)
    db.session.add(contents)
    db.session.commit()
    return users_schema.jsonify(contents)

@app.route('/update/<id>/', methods = ['PUT'])
def update_users(id):
    user = Users.query.get(id)
    title = request.json['title']
    body = request.json['body']
    user.title = title 
    user.body = body

    db.session.commit()
    return users_schema.jsonify(user)

@app.route('/delete/<id>/', methods = ['DELETE'])
def delete_users(id):
    user = Users.query.get(id)
    db.session.delete(user)
    db.session.commit()
    return users_schema.jsonify(user)

if __name__ == "__main__":
    app.run(debug=True)



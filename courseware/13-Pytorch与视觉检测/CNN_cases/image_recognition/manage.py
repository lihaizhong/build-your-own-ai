
from flask_script import Manager
from App import create_app

# 允许的图片格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = create_app()
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

manager = Manager(app)
#app = Flask(__name__)

if __name__ == '__main__':
    manager.run()
    #app.run()

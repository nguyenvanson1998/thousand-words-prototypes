env: venv/touchfile

venv/touchfile: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate; pip install -Ur requirements.txt
	touch venv/touchfile

travel: env
	. venv/bin/activate; streamlit run travel_demo.py

clean:
	rm -rf venv
	find -iname "*.pyc" -delete
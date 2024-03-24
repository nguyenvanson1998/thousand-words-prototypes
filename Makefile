env: venv/touchfile

venv/touchfile: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate; pip install -Ur requirements.txt
	touch venv/touchfile

howto: env
	. venv/bin/activate; streamlit run howto_demo.py

router: env
	. venv/bin/activate; streamlit run router_demo.py

travel: env
	. venv/bin/activate; streamlit run travel_demo.py

clean:
	rm -rf venv
	find -iname "*.pyc" -delete

create_env:
	@echo "Creating virtual environment..."
	@python -m venv env

activate_env:
	@echo "Activating virtual environment..."
	@if exist "env/Scripts/activate" (env\Scripts\activate) else (echo "Virtual environment not found!")

install_requirements:
	@echo "Installing requirements..."
	@pip install -r ./sdk/requirements.txt

install_local:
	@echo "Installing local packages..."
	@pip install -e ./sdk
function changebashpaths() {
	output="source path_config.sh"
	echo $output >> ~/.bashrc
  source ~/.bashrc
}

read -r -p "Are you sure you want to edit bashrc to export paths? (Y/N)" response
if [[ $response =~ ^[Yy]$ ]]
	then
		changebashpaths
fi
# Running 'source export_from_bashrc.sh' will add running path_config.sh directly to bashrc
# for automatic updates on booting up the linux terminal.

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
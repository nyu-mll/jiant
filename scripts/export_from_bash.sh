# Running 'source scripts/export_from_bash.sh' from the base jiant/ dir will add the job of
# running user_config.sh directly to the bash startup script (bashrc for Linux and bash_profile for MacOS).

function changebashpaths() {
	output="source $(pwd)/user_config.sh"
	case "$(uname -s)" in
		Darwin)
			echo $output >> ~/.bash_profile
			source ~/.bash_profile
		;;
		Linux)
			echo $output >> ~/.bashrc
			source ~/.bashrc
		;;
		*)
			echo 'Automatic path setup is only configured for MacOS and Linux.'
		;;
	esac
}

read -r -p "Are you sure you want to edit your bash configuration to set up jiant paths? (Y/N) " response
if [[ $response =~ ^[Yy]$ ]]
	then
		changebashpaths
fi

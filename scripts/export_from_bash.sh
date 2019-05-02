# Running 'source scripts/export_from_bash.sh' will add the job of running path_config.sh directly to bash
# (bashrc for Linux and bash_profile for Mac.)
# for automatic updates on booting up the terminals.

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
			echo 'not supported for non MAC/LINUX OS.'
		;;
	esac
}

read -r -p "Are you sure you want to edit bash file to export paths? (Y/N)" response
if [[ $response =~ ^[Yy]$ ]]
	then
		changebashpaths
fi

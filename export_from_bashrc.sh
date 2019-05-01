# Running 'source export_from_bashrc.sh' will add the job of running path_config.sh directly to bashrc
# for automatic updates on booting up the linux terminal.

function changebashpaths() {
	output="source path_config.sh"
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